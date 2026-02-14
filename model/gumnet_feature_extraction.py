import torch
import torch.nn as nn
import torch.nn.functional as F
from model.feature_l2_norm import FeatureL2Norm
from spectral import DCTSpectralPooling

class GumNetFeatureExtraction(nn.Module):
    """
    Core 2D Feature Extraction Module for the GumNet Siamese architecture.

    Args:
        in_channels (int, optional): Number of channels in the input tensor. 
            Defaults to 1 (expected for grayscale images like fingerprints).

    Shape:
        - Input: `(B, in_channels, 192, 192)` where `B` is the batch size.
        - Output: `(B, 512, 14, 14)` representing the L2-normalized deep feature maps.

    Architectural Flow & Tensor Dimensions:
        - Input:        (B, 1, 192, 192)
        - Block 1:
            - Conv2d:   (B, 32, 190, 190)
            - DCTPool:  (B, 32, 100, 100)  | Freq Mask: 85x85
        - Block 2:
            - Conv2d:   (B, 64, 98, 98)
            - DCTPool:  (B, 64, 50, 50)    | Freq Mask: 42x42
        - Block 3:
            - Conv2d:   (B, 128, 48, 48)
            - DCTPool:  (B, 128, 25, 25)   | Freq Mask: 21x21
        - Block 4:
            - Conv2d:   (B, 256, 23, 23)
            - DCTPool:  (B, 256, 16, 16)   | Freq Mask: 14x14
        - Block 5:
            - Conv2d:   (B, 512, 14, 14)
            - L2Norm:   (B, 512, 14, 14)   | Normalized across dim=1 (channels)

    Examples:
        >>> feature_extractor = GumNetFeatureExtraction(in_channels=1)
        >>> image_a = torch.randn(8, 1, 192, 192) # Batch of 8 images
        >>> image_b = torch.randn(8, 1, 192, 192) # Batch of 8 paired images
        >>> features_a = feature_extractor(image_a)
        >>> features_b = feature_extractor(image_b)
        >>> print(features_a.shape)
        torch.Size([8, 512, 14, 14])
    """


    def __init__(self, in_channels=1):
        super(GumNetFeatureExtraction, self).__init__()
        
        # Block 1: 192x192 -> Conv(32-3x3-1) -> 190x190 -> DCTSpectralPooling(85x85) -> 100x100
        self.shared_conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = DCTSpectralPooling(in_height=190, in_width=190, 
                                        freq_h=85, freq_w=85, 
                                        out_height=100, out_width=100)
        
        # Block 2: 100x100 -> Conv(64-3x3-1) -> 98x98 -> DCTSpectralPooling(42x42) -> 50x50
        self.shared_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = DCTSpectralPooling(in_height=98, in_width=98, 
                                        freq_h=42, freq_w=42, 
                                        out_height=50, out_width=50)
        
        # Block 3: 50x50 -> Conv(128-3x3-1) -> 48x48 -> DCTSpectralPooling(21x21) -> 25x25
        self.shared_conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = DCTSpectralPooling(in_height=48, in_width=48, 
                                        freq_h=21, freq_w=21, 
                                        out_height=25, out_width=25)
        
        # Block 4: 25x25 -> Conv(256-3x3-1) -> 23x23 -> DCTSpectralPooling(14x14) -> 16x16
        self.shared_conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = DCTSpectralPooling(in_height=23, in_width=23, 
                                        freq_h=14, freq_w=14, 
                                        out_height=16, out_width=16)
        
        # Block 5: 16x16 -> Conv(512-3x3-1) -> 14x14
        self.shared_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(512)
        
        # L2 Normalization before Correlation
        self.l2_norm = FeatureL2Norm(dim=1)


    def forward(self, x):
            
        # --- BLOCK 1 ---
        x = self.shared_conv1(x)               # [B, 32, 190, 190]
        x = F.relu(x)
        x = self.bn1(x)
        x = self.pool1(x)                      # [B, 32, 100, 100]
        
        # --- BLOCK 2 ---
        x = self.shared_conv2(x)               # [B, 64, 98, 98]
        x = F.relu(x)
        x = self.bn2(x)
        x = self.pool2(x)                      # [B, 64, 50, 50]
        
        # --- BLOCK 3 ---
        x = self.shared_conv3(x)               # [B, 128, 48, 48]
        x = F.relu(x)
        x = self.bn3(x)
        x = self.pool3(x)                      # [B, 128, 25, 25]
        
        # --- BLOCK 4 ---
        x = self.shared_conv4(x)               # [B, 256, 23, 23]
        x = F.relu(x)
        x = self.bn4(x)
        x = self.pool4(x)                      # [B, 256, 16, 16]
        
        # --- BLOCK 5 ---
        x = self.shared_conv5(x)               # [B, 512, 14, 14]
        x = self.bn5(x)
        x = self.l2_norm(x)                    # [B, 512, 14, 14]
        
        return x