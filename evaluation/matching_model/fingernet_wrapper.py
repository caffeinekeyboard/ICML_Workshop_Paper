import torch
from torch import nn
import torch.nn.functional as F
from evaluation.matching_model.fingernet import FingerNet
import numpy as np
import cv2

class FingerNetWrapper(nn.Module):
    def __init__(self, model: FingerNet, minutiae_threshold: float = 0.1, max_candidates: int = 100):
        super().__init__()
        self.fingernet = model
        self.default_minutiae_threshold = minutiae_threshold
        self.default_max_candidates = max_candidates

    def forward(
        self,
        x: torch.Tensor,
        minutiae_threshold: float | None = None,
        max_candidates: int | None = None,
    ) -> dict[str, torch.Tensor]:
        padded_x = self.preprocess(x)

        threshold = self.default_minutiae_threshold if minutiae_threshold is None else minutiae_threshold
        max_keep = self.default_max_candidates if max_candidates is None else max_candidates

        with torch.inference_mode():
            raw_outputs = self.fingernet(padded_x)

        post_x = self.postprocess(raw_outputs, threshold, max_keep)

        return post_x

    def prepare_input(self, x: np.ndarray) -> torch.Tensor:
        """Converts a numpy image to a torch tensor suitable for the model."""
        # Check if input is 2D (H, W)
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)  # add channel dimension
            x = np.expand_dims(x, axis=0)  # add batch dimension
        if x.ndim == 3:
            # This could be (C, H, W) or (B, H, W).
            # We assume (B, H, W) if B > 1
            if x.shape[0] > 1:
                x = np.expand_dims(x, axis=1)  # add channel dimension
            else:
                x = np.expand_dims(x, axis=0)  # add batch dimension
        if x.ndim == 4:
            tensor_x = torch.tensor(x, dtype=torch.float32)
        else:
            raise ValueError("Input numpy array must be 2D, 3D - with Channel, or 4D - with Batch.")
        
        # Detect device
        device = next(self.fingernet.parameters()).device
        tensor_x = tensor_x.to(device)
        return tensor_x

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        return F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)

    def postprocess(self, outputs: dict, threshold: float, max_candidates: int | None) -> dict[str, torch.Tensor]:
        return postprocess(outputs, threshold, max_candidates)

    def plot_minutiae(
        self,
        original_image: torch.Tensor,
        outputs: dict,
        save_path: str = 'minutiae_detection.png',
        denormalize: bool = True,
        mean: float = 0.5,
        std: float = 0.5,
    ):
        """
        Plot the original image and detected minutiae points using OpenCV.
        
        Args:
            original_image: Input tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
            outputs: Dictionary containing 'minutiae' and other outputs from forward()
            save_path: Path to save the visualization image
        """
        # Handle different input shapes
        if original_image.ndim == 4:
            img = original_image[0, 0].cpu().numpy()  # Take first batch, first channel
        elif original_image.ndim == 3:
            img = original_image[0].cpu().numpy()  # Take first channel
        elif original_image.ndim == 2:
            img = original_image.cpu().numpy()
        else:
            raise ValueError(f"Unexpected image shape: {original_image.shape}")
        
        # Convert to a displayable 0-255 uint8 image
        img = img.astype(np.float32)
        if denormalize and img.max() <= 1.5:
            # Handle inputs normalized with mean/std (e.g., [-1, 1]) or [0, 1]
            if img.min() < 0:
                img = img * std + mean
            img = np.clip(img, 0.0, 1.0)
            img_norm = (img * 255.0).astype(np.uint8)
        else:
            img_norm = np.clip(img, 0.0, 255.0).astype(np.uint8)
        
        # Get minutiae points (first batch if multiple)
        minutiae = outputs['minutiae'][0].cpu().numpy()
        
        # Create visualization (single image with minutiae)
        h, w = img_norm.shape
        canvas = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
        
        shown_count = 0
        if len(minutiae) > 0:
            for i in range(len(minutiae)):
                x = int(minutiae[i, 0])
                y = int(minutiae[i, 1])
                angle = float(minutiae[i, 2])

                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                shown_count += 1
                
                # Draw minutiae point
                cv2.circle(canvas, (x, y), 3, (0, 0, 255), -1)
                
                # Draw orientation line
                arrow_length = 15
                end_x = int(x + arrow_length * np.cos(angle))
                end_y = int(y + arrow_length * np.sin(angle))
                cv2.arrowedLine(canvas, (x, y), (end_x, end_y), (0, 255, 255), 1, tipLength=0.3)
        
        # Add compact text labels for small images (e.g., 192x192)
        label = f'Minutiae: {shown_count}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        margin = 4
        x0, y0 = 5, 5
        x1, y1 = x0 + text_w + 2 * margin, y0 + text_h + 2 * margin
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x0 + margin, y0 + text_h + margin),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Save the image
        cv2.imwrite(save_path, canvas)
        print(f"Minutiae visualization saved to {save_path}")
        
        return canvas

def postprocess(outputs: dict, threshold: float, max_candidates: int | None) -> dict[str, torch.Tensor]:
    # 1. Binarisation and cleaning of the segmentation mask
    cleaned_mask = _post_binarize_mask_fast(outputs['segmentation'])
    cleaned_mask_up = torch.nn.functional.interpolate(
        cleaned_mask.unsqueeze(1).float(),
        scale_factor=8,
        mode='nearest'
    ).squeeze(1)

    # 2. Detection of minutiae (including NMS)
    # The result is a list of tensors, one for each image in the batch.
    final_minutiae_list = _post_detect_minutiae(outputs, cleaned_mask, threshold, max_candidates)

    # 3. Processing of the orientation field
    ori = outputs['orientation']
    ori_idx = torch.argmax(ori, dim=1)
    ori_idx_up = torch.nn.functional.interpolate(
        ori_idx.unsqueeze(1).float(),
        scale_factor=8,
        mode='nearest'
    ).squeeze(1)
    orientation_field = (ori_idx_up * 2.0 - 89.) * torch.pi / 180.0
    orientation_field = orientation_field * cleaned_mask_up

    # 4. Enhanced image processing
    enh_real = outputs['enhanced_real'].squeeze(1)
    enh_real = enh_real * cleaned_mask_up
    
    # Min-Max normalisation for visualisation
    b, h, w = enh_real.shape
    enh_flat = enh_real.view(b, -1)
    enh_min = enh_flat.min(dim=1, keepdim=True)[0]
    enh_max = enh_flat.max(dim=1, keepdim=True)[0]
    enh_norm = (enh_flat - enh_min) / (enh_max - enh_min + 1e-8)
    enh_visual = (enh_norm.view(b, h, w) * 255).byte()

    return {
        'minutiae': final_minutiae_list, # type: ignore
        'enhanced_image': enh_visual,
        'segmentation_mask': (cleaned_mask_up * 255).byte(),
        'orientation_field': orientation_field
    }

def gaussian_blur_torch(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    def _get_gaussian_kernel1d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
        coords = torch.arange(kernel_size, device=device, dtype=dtype)
        coords -= kernel_size // 2
        # avoid using tensor ** operations (pow) which can trigger symbolic
        # interpretation inside torch.compile / torch._dynamo (sympy interp)
        coords_sq = coords * coords
        sigma_sq = sigma * sigma
        g = torch.exp(-(coords_sq) / (2 * sigma_sq))
        g /= g.sum()
        return g
    
    # 1. Obtain the 1D kernel
    kernel_1d = _get_gaussian_kernel1d(kernel_size, sigma, device=image.device, dtype=image.dtype)
    
    # 2. Get the number of channels to apply blur to each one independently
    B, C, H, W = image.shape
    
    # 3. Prepare kernels for horizontal and vertical convolution
    # conv2d expects a shape [out_channels, in_channels/groups, kH, kW]
    # We use `groups=C` so that each channel is convolved with its own kernel.
    kernel_h = kernel_1d.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
    kernel_v = kernel_1d.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)

    # 4. Calculate padding to maintain image size
    padding = kernel_size // 2
    
    # 5. Apply horizontal convolution
    blurred_h = F.conv2d(image, kernel_h, padding=(0, padding), groups=C)
    
    # 6. Apply vertical convolution on the result of horizontal convolution
    blurred_hv = F.conv2d(blurred_h, kernel_v, padding=(padding, 0), groups=C)
    
    return blurred_hv

def _post_binarize_mask_fast(seg_map: torch.Tensor) -> torch.Tensor:
    """
    Binarize and clean the segmentation mask extremely quickly,
    using a Gaussian blur implementation with PyTorch only.
    """
    # 1. Binarize the input mask to values 0.0 or 1.0
    binarized_float = torch.round(seg_map.squeeze(1))
    
    # 2. Adds the channel dimension to the convolution
    # Shape: [B, H, W] -> [B, 1, H, W]
    image_with_channel = binarized_float.unsqueeze(1)
    
    # 3. Applies fast Gaussian blur
    blurred = gaussian_blur_torch(image_with_channel, kernel_size=5, sigma=1.5)
    
    # 4. Re-bin the result to obtain the final, clean mask.
    cleaned_mask = torch.round(blurred)

    return cleaned_mask.squeeze(1)

def _post_detect_minutiae(
    outputs: dict,
    cleaned_mask: torch.Tensor,
    threshold: float,
    max_candidates: int | None,
) -> list:
    """Detects, filters, and applies NMS in detail for an entire batch."""
    mnt_score_batch = outputs['minutiae_score'].squeeze(1) * cleaned_mask
    mnt_orient_batch = outputs['minutiae_orientation']
    mnt_x_offset_batch = outputs['minutiae_x_offset']
    mnt_y_offset_batch = outputs['minutiae_y_offset']
    
    batch_size = mnt_score_batch.shape[0]
    final_minutiae_list = []

    for i in range(batch_size):
        # Find coordinates of details above the threshold
        rows, cols = torch.where(mnt_score_batch[i] > threshold)
        if rows.shape[0] == 0:
            final_minutiae_list.append(torch.empty((0, 4), device=mnt_score_batch.device))
            continue

        # Extract scores, angles, and offsets
        scores = mnt_score_batch[i][rows, cols]
        if max_candidates is not None and scores.numel() > max_candidates:
            topk = torch.topk(scores, k=max_candidates)
            keep_idx = topk.indices
            scores = topk.values
            rows = rows[keep_idx]
            cols = cols[keep_idx]
        angles_idx = torch.argmax(mnt_orient_batch[i, :, rows, cols], dim=0)
        x_offsets = torch.argmax(mnt_x_offset_batch[i, :, rows, cols], dim=0)
        y_offsets = torch.argmax(mnt_y_offset_batch[i, :, rows, cols], dim=0)
        
        # Calculate final values
        angles = (angles_idx * 2.0 - 89.0) * (torch.pi / 180.0)
        x_coords = cols * 8.0 + x_offsets
        y_coords = rows * 8.0 + y_offsets
        
        minutiae_raw = torch.stack([x_coords, y_coords, angles, scores], dim=-1)
        
        # Apply NMS
        final_minutiae = _post_nms(minutiae_raw)
        final_minutiae_list.append(final_minutiae)
        
    return final_minutiae_list

def _post_nms(minutiae: torch.Tensor, dist_thresh: float = 16.0, angle_thresh: float = torch.pi/6) -> torch.Tensor:
    """Applies Non-Maximum Suppression (NMS) to a minutiae tensor."""
    if minutiae.shape[0] == 0:
        return minutiae

    # Sort by score (descending)
    order = torch.argsort(minutiae[:, 3], descending=True)
    minutiae = minutiae[order]

    # Calculates Euclidean and angular distance matrix
    dist_matrix = torch.cdist(minutiae[:, :2], minutiae[:, :2])
    
    # Calculation of angular distance via broadcasting
    angles1 = minutiae[:, 2].unsqueeze(1) # [N, 1]
    angles2 = minutiae[:, 2].unsqueeze(0) # [1, N]
    angle_delta = torch.abs(angles1 - angles2)
    angle_matrix = torch.minimum(angle_delta, 2 * torch.pi - angle_delta)

    # Suppression mask: True where distance AND angle are less than threshold
    suppress_mask = (dist_matrix < dist_thresh) & (angle_matrix < angle_thresh)
    
    keep = torch.ones(minutiae.shape[0], dtype=torch.bool, device=minutiae.device)
    for i in range(minutiae.shape[0]):
        if keep[i]:
            # Delete all other points that are too close to this one.
            # torch.where returns a tuple, we take the first element
            suppress_indices = torch.where(suppress_mask[i, i+1:])[0]
            keep[i + 1 + suppress_indices] = False
            
    return minutiae[keep]