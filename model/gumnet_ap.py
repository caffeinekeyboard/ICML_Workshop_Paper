import torch.nn as nn
from model.gumnet_feature_extraction_ap import GumNetFeatureExtractionAP
from model.gumnet_siamese_matching_ap import GumNetSiameseMatchingAP
from model.gumnet_spatial_alignment_ap import GumNetSpatialAlignmentAP

class GumNetAP(nn.Module):
    """
    Complete GumNetAP architecture for fingerprint matching with spatial alignment.

    This model combines feature extraction, siamese correlation matching, and spatial alignment
    to perform fingerprint verification with geometric transformation prediction.

    Args:
        in_channels (int, optional): Number of input channels for images. Defaults to 1.

    Shape:
        - Input template: `(B, in_channels, 192, 192)`
        - Input impression: `(B, in_channels, 192, 192)`
        - Output warped_impression: `(B, in_channels, 192, 192)`
        - Output affine_matrix: `(B, 2, 3)`

    Examples:
        >>> model = GumNetAP()
        >>> template = torch.randn(4, 1, 192, 192)
        >>> impression = torch.randn(4, 1, 192, 192)
        >>> warped, matrix = model(template, impression)
        >>> print(warped.shape, matrix.shape)
        torch.Size([4, 1, 192, 192]) torch.Size([4, 2, 3])
    """

    def __init__(self, in_channels=1):
        super(GumNetAP, self).__init__()

        self.feature_extractor = GumNetFeatureExtractionAP(in_channels=in_channels)
        self.siamese_matcher = GumNetSiameseMatchingAP()
        self.spatial_aligner = GumNetSpatialAlignmentAP()

    def forward(self, template, impression):
        """
        Forward pass through the complete GumNetMP.

        Args:
            template (torch.Tensor): Template fingerprint image, shape (B, C, 192, 192)
            impression (torch.Tensor): Impression fingerprint image to be aligned, shape (B, C, 192, 192)

        Returns:
            warped_impression (torch.Tensor): Spatially aligned impression, shape (B, C, 192, 192)
            affine_matrix (torch.Tensor): Predicted affine transformation matrix, shape (B, 2, 3)
        """

        # Feature Extraction Module
        template_features = self.feature_extractor(template, branch='Sa')      # [B, 512, 19, 19]
        impression_features = self.feature_extractor(impression, branch='Sb')  # [B, 512, 19, 19]

        # Siamese Matching Module
        corr_ab, corr_ba = self.siamese_matcher(template_features, impression_features)  # [B, 50176] each

        # Spatial Alignment Module
        warped_impression, affine_matrix = self.spatial_aligner(corr_ab, corr_ba, impression) # [B, 1, 192, 192], [B, 2, 3]

        return warped_impression, affine_matrix
