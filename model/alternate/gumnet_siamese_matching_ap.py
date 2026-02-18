import torch.nn as nn
from model.layers.feature_correlation_2d import FeatureCorrelation2D
from model.layers.feature_l2_norm import FeatureL2Norm

class GumNetSiameseMatchingAP(nn.Module):
    """
    Siamese Feature Correlation and Regression Module for the 2D GumNetAP architecture.

    Args:
        in_channels (int, optional): Flattened spatial dimension of the input feature maps (H * W = 19 * 19 = 361).
            Defaults to 361.
        out_channels (int, optional): Number of filters in the regression convolutions. 
            Defaults to 1024.

    Shape:
        - Input v_a: `(B, 512, 19, 19)` where `B` is the batch size (Template features).
        - Input v_b: `(B, 512, 19, 19)` where `B` is the batch size (Impression features).
        - Output: A tuple of two tensors `(out_ab, out_ba)`, each of shape `(B, 102400)`.

    Architectural Flow & Tensor Dimensions:
        - Inputs:       v_a (B, 512, 19, 19) and v_b (B, 512, 19, 19)
        - Correlation:
            - C_ab:     (B, 361, 19, 19)     | L2-Normalized across channel dimension
            - C_ba:     (B, 361, 19, 19)     | L2-Normalized across channel dimension
        - Regression Block AB (for C_ab):
            - Conv2d:   (B, 1024, 17, 17)    | 3x3, padding=0 ('valid')
            - Conv2d:   (B, 1024, 15, 15)    | 3x3, padding=0 ('valid')
            - Conv2d:   (B, 1024, 13, 13)    | 3x3, padding=0 ('valid')
            - Conv2d:   (B, 1024, 11, 11)    | 3x3, padding=0 ('valid')
            - Conv2d:   (B, 1024, 9, 9)      | 3x3, padding=0 ('valid')
            - Conv2d:   (B, 1024, 7, 7)      | 3x3, padding=0 ('valid')
            - Flatten:  (B, 50176)           | 1024 * 7 * 7
        - Regression Block BA (for C_ba):
            - Same operations as AB, but with strictly independent weights -> (B, 50176)

    Examples:
        >>> siamese_matcher = GumNetSiameseMatchingAP(in_channels=361, out_channels=1024)
        >>> v_a = torch.randn(4, 512, 19, 19) # Batch of 4 template feature maps
        >>> v_b = torch.randn(4, 512, 19, 19) # Batch of 4 impression feature maps
        >>> out_ab, out_ba = siamese_matcher(v_a, v_b)
        >>> print(out_ab.shape, out_ba.shape)
        torch.Size([4, 50176]) torch.Size([4, 50176])
    """
    def __init__(self, in_channels: int = 361, out_channels: int = 1024):
        """
        """
        super(GumNetSiameseMatchingAP, self).__init__()
        
        self.correlation_layer = FeatureCorrelation2D()
        self.regression_block_ab = nn.Sequential(
            
            # Block 1: 19x19 -> Conv(1024-3x3-1-'valid') -> 17x17
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 2: 17x17 -> Conv(1024-3x3-1-'valid') -> 15x15
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 3: 15x15 -> Conv(1024-3x3-1-'valid') -> 13x13
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 4: 13x13 -> Conv(1024-3x3-1-'valid') -> 11x11
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 5: 11x11 -> Conv(1024-3x3-1-'valid') -> 9x9
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 6: 9x9 -> Conv(1024-3x3-1-'valid') -> 7x7
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Flatten: (B, 1024, 7, 7) -> (B, 50176)
            nn.Flatten()
        )
        self.regression_block_ba = nn.Sequential(
            
            # Block 1: 19x19 -> Conv(1024-3x3-1-'valid') -> 17x17
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 2: 17x17 -> Conv(1024-3x3-1-'valid') -> 15x15
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 3: 15x15 -> Conv(1024-3x3-1-'valid') -> 13x13
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 4: 13x13 -> Conv(1024-3x3-1-'valid') -> 11x11
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 5: 11x11 -> Conv(1024-3x3-1-'valid') -> 9x9
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Block 6: 9x9 -> Conv(1024-3x3-1-'valid') -> 7x7
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # Flatten: (B, 1024, 7, 7) -> (B, 50176)
            nn.Flatten()
        )
        self.l2_norm = FeatureL2Norm()

    def forward(self, v_a, v_b):
        
        # --- BLOCK FOR CORRELATION MAP Cab ---
        c_ab = self.correlation_layer(v_a, v_b)
        c_ab = self.l2_norm(c_ab)
        out_ab = self.regression_block_ab(c_ab)
        
        # --- BLOCK FOR CORRELATION MAP Cba ---
        c_ba = self.correlation_layer(v_b, v_a)
        c_ba = self.l2_norm(c_ba)
        out_ba = self.regression_block_ba(c_ba)
        
        return out_ab, out_ba