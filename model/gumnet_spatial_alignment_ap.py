import torch
import torch.nn as nn
import torch.nn.functional as F

class GumNetSpatialAlignmentAP(nn.Module):
    """
    Core 2D Transformation Regression and Spatial Warping Module for the GumNetAP architecture.

    Args:
        input_dim (int, optional): Total number of features after concatenating 
            both bidirectional correlation vectors. Defaults to 100352 (expected 
            for two 50176-dimensional correlation vectors).

    Shape:
        - Input 1 (c_ab): `(B, D)` where `B` is the batch size and `D` is the dimension of a single correlation vector (e.g., 50176).
        - Input 2 (c_ba): `(B, D)` matching `c_ab`.
        - Input 3 (source_image): `(B, C, H, W)` where `C` is channels, `H` is height, `W` is width.
        - Output 1 (warped_image): `(B, C, H, W)` representing the spatially aligned source image.
        - Output 2 (affine_matrix): `(B, 2, 3)` representing the predicted 2D affine transformation.

    Architectural Flow & Tensor Dimensions:
        - Inputs:
            - c_ab:                 (B, 50176)
            - c_ba:                 (B, 50176)
            - source_image:         (B, 1, 192, 192)
        - Concatenation:            (B, 100352)  | torch.cat([c_ab, c_ba], dim=1)
        - Block 1 (Thinking):
            - Linear + ReLU:        (B, 2000)
        - Block 2 (Thinking):
            - Linear + ReLU:        (B, 2000)
        - Block 3 (Decision Maker):
            - Linear + Sigmoid:     (B, 3)     | Range: [0, 1]
        - Geometric Scaling:
            - Theta (Rotation):     [-pi, pi] radians
            - Tx, Ty (Translation): [-1.0, 1.0] normalized coordinates
        - Affine Matrix Generation: (B, 2, 3)  | Identity initialized at Epoch 0
        - Spatial Transformer:      (B, 1, 192, 192) | Bilinear grid sampling

    Examples:
        >>> regressor = TransformationRegressor2D(input_dim=100352)
        >>> c_ab = torch.randn(8, 50176) # Correlation of A to B
        >>> c_ba = torch.randn(8, 50176) # Correlation of B to A
        >>> image_a = torch.randn(8, 1, 192, 192) # Source fingerprint
        >>> warped_image, affine_matrix = regressor(c_ab, c_ba, image_a)
        >>> print(warped_image.shape)
        torch.Size([8, 1, 192, 192])
        >>> print(affine_matrix.shape)
        torch.Size([8, 2, 3])
    """
    def __init__(self, input_dim=100352):
        super(GumNetSpatialAlignmentAP, self).__init__()
        
        # Fully Connected Layer 1: [B, 100352] -> [B, 2000]
        self.fc1 = nn.Linear(input_dim, 2000)
        
        # Fully Connected Layer 2: [B, 2000] -> [B, 2000]
        self.fc2 = nn.Linear(2000, 2000)
        
        # Fully Connected Layer 3: [B, 2000] -> [B, 3]
        self.fc_out = nn.Linear(2000, 3)
        
        # Initialize the final regressor head to predict identity transformation.
        self._initialize_identity()

    def _initialize_identity(self):
        nn.init.zeros_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, c_ab, c_ba, source_image):
        """
        Forward pass to predict the affine transformation parameters and warp the source image.
        
        Args:
            c_ab (torch.Tensor): Correlation features from branch A to B, shape (B, 50176).
            c_ba (torch.Tensor): Correlation features from branch B to A, shape (B, 50176).
            source_image (torch.Tensor): The fingerprint impression to be warped, shape (B, Channels, H, W).
            
        Returns:
            warped_image (torch.Tensor): The spatially aligned impression, shape (B, Channels, H, W).
            affine_matrix (torch.Tensor): The predicted affine transformation parameters, shape (B, 2, 3).
        """
        
        # Concatenate correlation features
        c = torch.cat([c_ab, c_ba], dim=1)     # [B, 100352]
        
        # Multilayer Perceptron 
        c = F.relu(self.fc1(c))
        c = F.relu(self.fc2(c))                # [B, 2000]
        
        # Final Regressor Head
        params = torch.sigmoid(self.fc_out(c)) # [B, 3] -> (theta, tx, ty) normalized to [0, 1]
        
        # Convert normalized parameters to actual transformation values
        theta = params[:, 0] * 2 * torch.pi - torch.pi
        tx = params[:, 1] * 2.0 - 1.0
        ty = params[:, 2] * 2.0 - 1.0
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        B = c.size(0)
        
        # Construct the affine transformation matrix.
        affine_matrix = torch.zeros(B, 2, 3, device=c.device)
        
        # Row 1: [cos(theta), -sin(theta), tx]
        affine_matrix[:, 0, 0] = cos_t
        affine_matrix[:, 0, 1] = -sin_t
        affine_matrix[:, 0, 2] = tx
        
        # Row 2: [sin(theta), cos(theta), ty]
        affine_matrix[:, 1, 0] = sin_t
        affine_matrix[:, 1, 1] = cos_t
        affine_matrix[:, 1, 2] = ty
        
        # Apply Spatial Transformer
        grid = F.affine_grid(affine_matrix, source_image.size(), align_corners=False)
        warped_image = F.grid_sample(source_image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return warped_image, affine_matrix