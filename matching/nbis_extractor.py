import torch
import nbis
import torch.nn as nn
from nbis import NbisExtractorSettings
import numpy as np
from PIL import Image
import io

# Configuration for the NbisExtractor
settings = NbisExtractorSettings(
    # Do not filter on minutiae quality (get all minutiae)
    min_quality=0.0,
    # Do not get the fingerprint center or ROI
    get_center=False,
    # Do not use SIVV to check if the image is a fingerprint
    check_fingerprint=False,
    # Compute the NFIQ2 quality score
    compute_nfiq2=False,
    # No specific PPI, use the default
    ppi=None,
)

class Nbis(nn.Module):
    def __init__(self, settings: NbisExtractorSettings = settings):
        super().__init__()
        self.extractor = nbis.new_nbis_extractor(settings)
    
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:  # image: (B,1,H,W) or (1,H,W) or (H,W)
        if image.dim() == 2:
            batch = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            batch = image.unsqueeze(0)
        else:
            batch = image

        minutiae_list = []
        batch_size, _, height, width = batch.shape
        device = image.device
        
        for img in batch:
            img = img.squeeze(0).detach().cpu()
            if img.dtype != torch.uint8:
                if img.min() < 0:
                    img = img * 0.5 + 0.5
                img = (img.clamp(0, 1) * 255).to(torch.uint8)
            img_np = img.contiguous().numpy()
            pil_img = Image.fromarray(img_np, mode="L")
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            
            # Extract minutiae using NBIS
            nbis_result = self.extractor.extract_minutiae(image_bytes)
            points = nbis_result.get()
            
            # Convert to tensor format: [x, y, angle_radians, reliability]
            if len(points) > 0:
                minutiae_data = []
                for point in points:
                    x = float(point.x())
                    y = float(point.y())
                    angle_deg = float(point.angle())
                    angle_rad = np.radians(angle_deg)
                    reliability = float(point.reliability())
                    minutiae_data.append([x, y, angle_rad, reliability])
                
                minutiae_tensor = torch.tensor(minutiae_data, dtype=torch.float32, device=device)
            else:
                minutiae_tensor = torch.empty((0, 4), dtype=torch.float32, device=device)
            
            minutiae_list.append(minutiae_tensor)
        
        # Create orientation field (NBIS doesn't provide dense field, so create empty)
        orientation_field = torch.zeros((batch_size, height, width), dtype=torch.float32, device=device)
        
        return {
            'minutiae': minutiae_list, # type: ignore
            'orientation_field': orientation_field
        }