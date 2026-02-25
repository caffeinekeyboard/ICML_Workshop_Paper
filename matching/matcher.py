import torch
import torch.nn as nn
import torch.nn.functional as F


class matcher(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module, metric, performance, mask: bool = False, gumnet: bool = False):
        super().__init__()
        self.alignment = model1
        self.extractor = model2
        self.metric = metric
        self.performance = performance
        self.mask = mask
        self.gumnet = gumnet
        self._last_intermediates = None

    @staticmethod
    def _detach_tree(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {k: matcher._detach_tree(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            container = [matcher._detach_tree(v) for v in value]
            return type(value)(container)
        return value
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, orig_img1: torch.Tensor, orig_img2: torch.Tensor) -> torch.Tensor:
        if self.gumnet:
            _, control_points = self.alignment(img1, img2) # type: ignore
            alignment1 = warp_original(orig_img1, control_points)
        else:
            alignment1 = self.alignment(img1, img2)
        print("Alignment done.")
        extractor1 = self.extractor(alignment1)
        extractor2 = self.extractor(orig_img2)
        print("Feature extraction done.")
        score = self.metric(alignment1, orig_img2, extractor1, extractor2, alpha=1.0, beta=0.0, delta=0.0, mask=self.mask)

        self._last_intermediates = {
            "alignment1": self._detach_tree(alignment1),
            "origninal_img": self._detach_tree(orig_img2),
            "extractor1": self._detach_tree(extractor1),
            "extractor2": self._detach_tree(extractor2),
        }
        
        performance = self.performance(score)
        return performance

    def save_intermediates(self, file_path: str) -> None:
        if self._last_intermediates is None:
            raise RuntimeError("No intermediates cached. Run forward() first.")
        def to_cpu(tree):
            if torch.is_tensor(tree):
                return tree.detach().cpu()
            if isinstance(tree, dict):
                return {k: to_cpu(v) for k, v in tree.items()}
            if isinstance(tree, (list, tuple)):
                container = [to_cpu(v) for v in tree]
                return type(tree)(container)
            return tree

        cpu_copy = {k: to_cpu(v) for k, v in self._last_intermediates.items()}
        torch.save(cpu_copy, file_path)

    @staticmethod
    def load_intermediates(file_path: str) -> dict:
        return torch.load(file_path, map_location="cpu")
    
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