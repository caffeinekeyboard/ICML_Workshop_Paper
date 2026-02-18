import torch.nn as nn
import torch.nn.functional as F
from model.layers.feature_l2_norm import FeatureL2Norm

class GumNetFeatureExtractionAP(nn.Module):
    """
    Core 2D Feature Extraction Module for the GumNet Siamese architecture with AvgPooling.

    Args:
        in_channels (int, optional): Number of channels in the input tensor. 
            Defaults to 1 (expected for grayscale images like fingerprints).

    Shape:
        - Input: `(B, in_channels, 192, 192)` where `B` is the batch size.
        - Output: `(B, 512, 42, 42)` representing the L2-normalized deep feature maps.

    Architectural Flow & Tensor Dimensions:
        - Input:        (B, 1, 192, 192)
        - Block 1:
            - Conv2d:   (B, 32, 190, 190)
            - AvgPool:  (B, 32, 95, 95)
        - Block 2:
            - Conv2d:   (B, 64, 93, 93)
        - Block 3:
            - Conv2d:   (B, 128, 91, 91)
            - AvgPool:  (B, 128, 45, 45)
        - Block 4:
            - Conv2d:   (B, 256, 43, 43)
            - AvgPool:  (B, 256, 21, 21)
        - Block 5:
            - Conv2d:   (B, 512, 19, 19)
            - L2Norm:   (B, 512, 19, 19)   | Normalized across dim=1 (channels)

    Examples:
        >>> feature_extractor = GumNetFeatureExtractionAP(in_channels=1)
        >>> image_a = torch.randn(8, 1, 192, 192) # Batch of 8 images
        >>> image_b = torch.randn(8, 1, 192, 192) # Batch of 8 paired images
        >>> features_a = feature_extractor(image_a, branch='Sa')
        >>> features_b = feature_extractor(image_b, branch='Sb')
        >>> print(features_a.shape)
        torch.Size([8, 512, 19, 19])
    """


    def __init__(self, in_channels=1):
        super(GumNetFeatureExtractionAP, self).__init__()
        
        # Block 1: 192x192 -> Conv(32-3x3-1-'valid') -> 190x190 -> AvgPooling(2x2x2) -> 95x95
        self.shared_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.bn1_sa = nn.BatchNorm2d(32)
        self.bn1_sb = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Block 2: 95x95 -> Conv(64-3x3-1-'valid') -> 93x93
        self.shared_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2_sa = nn.BatchNorm2d(64)
        self.bn2_sb = nn.BatchNorm2d(64)
        
        # Block 3: 93x93 -> Conv(128-3x3-1-'valid') -> 91x91 -> AvgPooling(2x2x2) -> 45x45
        self.shared_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn3_sa = nn.BatchNorm2d(128)
        self.bn3_sb = nn.BatchNorm2d(128)
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Block 4: 45x45 -> Conv(256-3x3-1-'valid') -> 43x43 -> AvgPooling(2x2x2) -> 21x21
        self.shared_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn4_sa = nn.BatchNorm2d(256)
        self.bn4_sb = nn.BatchNorm2d(256)
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Block 5: 21x21 -> Conv(512-3x3-1-'valid') -> 19x19
        self.shared_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.bn5_sa = nn.BatchNorm2d(512)
        self.bn5_sb = nn.BatchNorm2d(512)
        
        # L2 Normalization before Correlation
        self.l2_norm = FeatureL2Norm()


    def forward(self, x, branch):
        """
        Forward pass through the GumNetAP feature extraction module.

        Args:
            x (torch.Tensor): Input image tensor of shape `(B, C, H, W)`.
            branch (str): Either 'Sa' or 'Sb', indicating which branch of the network to use.

        Returns:
            torch.Tensor: Output feature map tensor of shape `(B, 512, 14, 14)`.
        """
            
        # --- BLOCK 1 ---
        x = self.shared_conv1(x)               # [B, 32, 190, 190]
        x = F.relu(x)
        if branch == 'Sa':
            x = self.bn1_sa(x)
        elif branch == 'Sb':
            x = self.bn1_sb(x)
        else:
            raise ValueError("branch must be 'Sa' or 'Sb'")
        x = self.pool1(x)                      # [B, 32, 95, 95]
        
        # --- BLOCK 2 ---
        x = self.shared_conv2(x)               # [B, 64, 93, 93]
        x = F.relu(x)
        if branch == 'Sa':
            x = self.bn2_sa(x)
        elif branch == 'Sb':
            x = self.bn2_sb(x)
        
        # --- BLOCK 3 ---
        x = self.shared_conv3(x)               # [B, 128, 91, 91]
        x = F.relu(x)
        if branch == 'Sa':
            x = self.bn3_sa(x)
        elif branch == 'Sb':
            x = self.bn3_sb(x)
        x = self.pool3(x)                      # [B, 128, 45, 45]
        
        # --- BLOCK 4 ---
        x = self.shared_conv4(x)               # [B, 256, 43, 43]
        x = F.relu(x)
        if branch == 'Sa':
            x = self.bn4_sa(x)
        elif branch == 'Sb':
            x = self.bn4_sb(x)
        x = self.pool4(x)                      # [B, 256, 21, 21]
        
        # --- BLOCK 5 ---
        x = self.shared_conv5(x)               # [B, 512, 19, 19]
        if branch == 'Sa':
            x = self.bn5_sa(x)
        elif branch == 'Sb':
            x = self.bn5_sb(x)
        x = self.l2_norm(x)                    # [B, 512, 19, 19]
        
        return x