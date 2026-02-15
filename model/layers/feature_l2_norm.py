import torch
import torch.nn as nn

class FeatureL2Norm(nn.Module):
    """
    Normalizes features using mathematically exact L2 norm with epsilon integration prior to square root, matching the xulabs/aitom TensorFlow implementation.
    
    References
    ----------
    [1]  Convolutional neural network architecture for geometric matching, Ignacio Rocco, et al.
    """
    def __init__(self, eps=1e-6):
        super(FeatureL2Norm, self).__init__()
        self.eps = eps

    def forward(self, feature):
        squared_sum = torch.sum(torch.pow(feature, 2), dim=1, keepdim=True)
        norm = torch.pow(squared_sum + self.eps, 0.5)

        return torch.div(feature, norm)