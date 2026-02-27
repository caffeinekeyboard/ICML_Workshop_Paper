import torch
import torch.nn.functional as F


def warp_original(img_tensor: torch.Tensor, control_points: torch.Tensor):

    dense_flow = F.interpolate(
            control_points, 
            size=(640, 640), 
            mode='bicubic', 
            align_corners=True
        )           
    
    dense_flow = dense_flow.permute(0, 2, 3, 1)

    y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 640, device='cpu'),
            torch.linspace(-1.0, 1.0, 640, device='cpu'),
            indexing='ij'
        )
    
    base_grid = torch.stack([x, y], dim=-1)
    deformation_grid = base_grid + dense_flow

    warped_image = F.grid_sample(
        img_tensor, 
        deformation_grid, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )

    return warped_image