from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile
from typing import Any, Mapping, Union

import torch
import torch.nn as nn
from model.gumnet_feature_extraction import GumNetFeatureExtraction
from model.gumnet_siamese_matching import GumNetSiameseMatching
from model.gumnet_non_linear_alignment import GumNetNonLinearAlignment

class GumNet(nn.Module):
    """
    Complete GumNet architecture for fingerprint matching with non-linear spatial alignment.

    Args:
        in_channels (int, optional): Number of input channels for images. Defaults to 1.

    Shape:
        - Input template: `(B, in_channels, 192, 192)`
        - Input impression: `(B, in_channels, 192, 192)`
        - Output warped_impression: `(B, in_channels, 192, 192)`
        - Output control_points: `(B, 2, grid_size, grid_size)` (default grid_size=4)

    Notes:
        The spatial alignment module returns the warped impression and the predicted
        control-point displacements (not an affine matrix). The default `grid_size`
        is 4, so the control points shape is `(B, 2, 4, 4)` unless configured
        otherwise in `GumNetNonLinearAlignment`.

    Examples:
        >>> model = GumNet()
        >>> template = torch.randn(4, 1, 192, 192)
        >>> impression = torch.randn(4, 1, 192, 192)
        >>> warped, control_points = model(template, impression)
        >>> print(warped.shape, control_points.shape)
        torch.Size([4, 1, 192, 192]) torch.Size([4, 2, 4, 4])
    """

    def __init__(self, in_channels=1, grid_size=4):
        super(GumNet, self).__init__()

        self.feature_extractor = GumNetFeatureExtraction(in_channels=in_channels)
        self.siamese_matcher = GumNetSiameseMatching()
        self.spatial_aligner = GumNetNonLinearAlignment(grid_size=grid_size)

    @staticmethod
    def _resolve_weights_root(weights_path: Path) -> Path:
        if weights_path.is_file():
            return weights_path
        if weights_path.is_dir():
            if (weights_path / "data.pkl").exists():
                return weights_path
            data_pkl_files = list(weights_path.rglob("data.pkl"))
            if len(data_pkl_files) == 1:
                return data_pkl_files[0].parent
        return weights_path

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint
            )
        else:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            return {k.replace("module.", ""): v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def _zip_torch_weights_dir(weights_root: Path) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="gumnet_weights_"))
        output_path = tmp_dir / f"{weights_root.name}.pth"
        base_folder = weights_root.name
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in weights_root.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(weights_root).as_posix()
                    arcname = f"{base_folder}/{rel_path}"
                    zip_info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                    zip_info.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zip_info, file_path.read_bytes())
        return output_path

    @classmethod
    def _load_state_dict_from_path(
        cls,
        weights_path: Path,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Mapping[str, Any]:
        resolved_path = cls._resolve_weights_root(weights_path)
        if resolved_path.is_dir():
            resolved_path = cls._zip_torch_weights_dir(resolved_path)
        checkpoint = torch.load(resolved_path, map_location=map_location)
        return cls._extract_state_dict(checkpoint)

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        map_location = kwargs.pop("map_location", "cpu")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        if isinstance(state_dict, (str, Path)):
            state_dict = self._load_state_dict_from_path(
                Path(state_dict),
                map_location=map_location,
            )
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, template, impression):
        """
        Forward pass through the complete GumNet.

        Args:
            template (torch.Tensor): Template fingerprint image, shape (B, C, 192, 192)
            impression (torch.Tensor): Impression fingerprint image to be aligned, shape (B, C, 192, 192)

        Returns:
            warped_impression (torch.Tensor): Spatially aligned impression, shape (B, C, 192, 192)
            control_points (torch.Tensor): Predicted control-point displacements, shape (B, 2, grid_size, grid_size).
        """

        # Feature Extraction Module
        template_features = self.feature_extractor(template, branch='Sa')      # [B, 512, 14, 14]
        impression_features = self.feature_extractor(impression, branch='Sb')  # [B, 512, 14, 14]

        # Siamese Matching Module
        corr_ab, corr_ba = self.siamese_matcher(template_features, impression_features)  # [B, 4096] each

        # Spatial Alignment Module
        warped_impression, control_points = self.spatial_aligner(corr_ab, corr_ba, impression) # [B, 1, 192, 192], [B, 2, grid_size, grid_size]

        return warped_impression, control_points
