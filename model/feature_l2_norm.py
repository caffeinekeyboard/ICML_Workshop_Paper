import torch.nn as nn
import torch.nn.functional as F

class FeatureL2Norm(nn.Module):
    """
    Normalizes the feature maps across the channel dimension.
    Equivalent to the Custom FeatureL2Norm layer in the 3D GumNet.
    """
    def __init__(self, dim=1):
        super(FeatureL2Norm, self).__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)