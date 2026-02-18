import torch
import torch.nn as nn
import torch.nn.functional as F

class GumNetNonLinearAlignment(nn.Module):
    """
    Non-Linear Transformation Regression and Spatial Warping Module for the GumNet architecture.

    Args:
        input_dim (int, optional): Total number of features after concatenating both bidirectional correlation vectors. 
            Defaults to 8192 (expected for two 4096-dimensional correlation vectors).
        grid_size (int, optional): Size of the control point grid for non-linear deformation.
            Defaults to 4, resulting in a 4x4 grid of control points (16 total).

    Shape:
        - Input 1 (c_ab): `(B, D)` where `B` is the batch size and `D` is the dimension of a single correlation vector (e.g., 4096).
        - Input 2 (c_ba): `(B, D)` matching `c_ab`.
        - Input 3 (source_image): `(B, C, H, W)` where `C` is channels, `H` is height, `W` is width.
        - Output 1 (warped_image): `(B, C, H, W)` representing the non-linearly aligned source image.
        - Output 2 (control_points): `(B, 2, grid_size, grid_size)` representing the predicted displacement vectors for each control point.

    Architectural Flow & Tensor Dimensions:
        - Inputs:
            - c_ab:                     (B, 4096)
            - c_ba:                     (B, 4096)
            - source_image:             (B, 1, 192, 192)
        - Concatenation:                (B, 8192)  | torch.cat([c_ab, c_ba], dim=1)
        - Block 1 (Thinking):
            - Linear + ReLU:            (B, 2000)
        - Block 2 (Thinking):
            - Linear + ReLU:            (B, 2000)
        - Block 3 (Control Point Predictor):
            - Linear:                   (B, grid_size * grid_size * 2) | For grid_size=4: (B, 32)
        - Reshape:                      (B, 2, grid_size, grid_size) | (B, 2, 4, 4)
        - Bicubic Upsampling:           (B, 2, 192, 192) | Dense flow field
        - Permute:                      (B, 192, 192, 2) | For grid_sample compatibility
        - Base Grid Creation:           (B, 192, 192, 2) | Normalized coordinates [-1, 1]
        - Deformation Grid:             (B, 192, 192, 2) | Base grid + dense flow
        - Spatial Transformer:          (B, 1, 192, 192) | Bilinear grid sampling

    Examples:
        >>> non_linear_regressor = GumNetNonLinearAlignment(input_dim=8192, grid_size=4)
        >>> c_ab = torch.randn(8, 4096) # Correlation of A to B
        >>> c_ba = torch.randn(8, 4096) # Correlation of B to A
        >>> image_a = torch.randn(8, 1, 192, 192) # Source fingerprint
        >>> warped_image, control_points = non_linear_regressor(c_ab, c_ba, image_a)
        >>> print(warped_image.shape)
        torch.Size([8, 1, 192, 192])
        >>> print(control_points.shape)
        torch.Size([8, 2, 4, 4])
    """
    def __init__(self, input_dim=8192, grid_size=4):
        super(GumNetNonLinearAlignment, self).__init__()
        
        self.grid_size = grid_size
        self.num_control_points = grid_size * grid_size
        
        # Fully Connected Layer 1: [B, 8192] -> [B, 2000]
        self.fc1 = nn.Linear(input_dim, 2000)
        
        # Fully Connected Layer 2: [B, 2000] -> [B, 2000]
        self.fc2 = nn.Linear(2000, 2000)
        
        # Fully Connected Layer 3: [B, 2000] -> [B, num_control_points * 2]
        self.fc_out = nn.Linear(2000, self.num_control_points * 2)
        
        self._initialize_identity()

    def _initialize_identity(self):
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _create_base_grid(self, B, H, W, device):
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing='ij'
        )
        base_grid = torch.stack([x, y], dim=-1)                 # (H, W, 2)
        return base_grid.unsqueeze(0).expand(B, -1, -1, -1)     # (B, H, W, 2)

    def forward(self, c_ab, c_ba, source_image):
        """
        Forward pass to predict the non-linear deformation field and warp the source image.
        
        Args:
            c_ab (torch.Tensor): Correlation features from branch A to B, shape (B, 4096).
            c_ba (torch.Tensor): Correlation features from branch B to A, shape (B, 4096).
            source_image (torch.Tensor): The fingerprint impression to be warped, shape (B, Channels, H, W).
            
        Returns:
            warped_image (torch.Tensor): The non-linearly aligned impression, shape (B, Channels, H, W).
            control_points (torch.Tensor): The predicted displacement vectors at control points, shape (B, 2, grid_size, grid_size).
        """
        B, C, H, W = source_image.size()
        device = source_image.device
        
        # Concatenate correlation features
        c = torch.cat([c_ab, c_ba], dim=1)     # [B, 8192]
        
        # Multilayer Perceptron 
        c = F.relu(self.fc1(c))
        c = F.relu(self.fc2(c))                # [B, 2000]               
        
        # Predict control point shifts
        raw_shifts = self.fc_out(c)            # [B, num_control_points * 2]
        
        # Reshape to spatial grid [B, 2, 4, 4]
        control_points = raw_shifts.view(B, 2, self.grid_size, self.grid_size)
        
        # Create the Dense Flow. 
        # Upsampling: grid_size x grid_size -> 192x192
        dense_flow = F.interpolate(
            control_points, 
            size=(H, W), 
            mode='bicubic', 
            align_corners=True
        )                                      # [B, 2, 192, 192]
        
        # Convert to [B, 192, 192, 2] for grid_sample expected format
        dense_flow = dense_flow.permute(0, 2, 3, 1)
        
        # Add the predicted non-linear flow to the base coordinate grid
        base_grid = self._create_base_grid(B, H, W, device)
        deformation_grid = base_grid + dense_flow
        
        # Apply Non-linear Spatial Transformer
        warped_image = F.grid_sample(
            source_image, 
            deformation_grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        
        return warped_image, control_points