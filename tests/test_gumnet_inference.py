from __future__ import annotations
import zipfile
from pathlib import Path

import pytest
import torch
import torchvision

from datasets.no_split_dataloader import get_no_split_dataloader
from model.gumnet import GumNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    "/home/marius/Asus/CVPR_Workshop/ICML_Workshop_Paper/data/FCV/FVC2004/Dbs/DB1_A"
)
WEIGHTS_PATH = PROJECT_ROOT / "model" / "gumnet_2d_best_noise_level_0_8x8_200.pth"


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


def _zip_torch_weights_dir(weights_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_gumnet(weights_file: Path) -> GumNet:
    checkpoint = torch.load(weights_file, map_location="cpu")
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    else:
        state_dict = checkpoint
    if isinstance(state_dict, dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    grid_size = 4
    if isinstance(state_dict, dict) and "spatial_aligner.fc_out.weight" in state_dict:
        out_features = state_dict["spatial_aligner.fc_out.weight"].shape[0]
        grid_size = int((out_features / 2) ** 0.5)

    model = GumNet(grid_size=grid_size)
    model.load_state_dict(state_dict, strict=True)
    return model


@pytest.fixture(scope="session")
def weights_file_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    weights_root = _resolve_weights_root(WEIGHTS_PATH)
    if weights_root.is_file():
        return weights_root

    if not weights_root.exists():
        pytest.skip(f"Weights not found at {WEIGHTS_PATH}")

    tmp_dir = tmp_path_factory.mktemp("gumnet_weights")
    zipped_path = tmp_dir / "gumnet_weights.pth"
    return _zip_torch_weights_dir(weights_root, zipped_path)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def model(weights_file_path: Path, device: torch.device) -> GumNet:
    model = _load_gumnet(weights_file_path)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def dataloader() -> torch.utils.data.DataLoader:
    if not DATA_ROOT.exists():
        pytest.skip(f"Dataset path not found: {DATA_ROOT}")

    loader = get_no_split_dataloader(
        data_root=str(DATA_ROOT),
        batch_size=1,
        num_workers=0,
    )
    if len(loader.dataset) == 0:
        pytest.skip("No samples found in the dataset.")
    return loader


def test_single_image_inference(model: GumNet, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
    batch = next(iter(dataloader))
    template = batch["Sa"].to(device)
    impression = batch["Sb"].to(device)

    with torch.no_grad():
        warped_impression, control_points = model(template, impression)

    assert warped_impression.shape[0] == 1
    assert warped_impression.shape[2:] == (192, 192)
    assert control_points.shape[0] == 1
    assert control_points.shape[1] == 2


def test_full_dataset_inference(model: GumNet, device: torch.device) -> None:
    if not DATA_ROOT.exists():
        pytest.skip(f"Dataset path not found: {DATA_ROOT}")

    loader = get_no_split_dataloader(
        data_root=str(DATA_ROOT),
        batch_size=8,
        num_workers=0,
    )
    if len(loader.dataset) == 0:
        pytest.skip("No samples found in the dataset.")

    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            template = batch["Sa"].to(device)
            impression = batch["Sb"].to(device)
            warped_impression, control_points = model(template, impression)
            assert warped_impression.shape[0] == template.shape[0]
            assert control_points.shape[0] == template.shape[0]
            total_samples += template.shape[0]

    assert total_samples == len(loader.dataset)


def test_save_warped_impression_output(
    model: GumNet,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> None:
    batch = next(iter(dataloader))
    template = batch["Sa"].to(device)
    impression = batch["Sb"].to(device)

    with torch.no_grad():
        warped_impression, _ = model(template, impression)

    output_dir = PROJECT_ROOT / "tests" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "template_warped_side_by_side.png"

    template_normalized = (template + 1.0) / 2.0
    template_normalized = template_normalized.clamp(0.0, 1.0).cpu()
    warped_normalized = (warped_impression + 1.0) / 2.0
    warped_normalized = warped_normalized.clamp(0.0, 1.0).cpu()

    side_by_side = torch.cat([template_normalized, warped_normalized], dim=-1)
    torchvision.utils.save_image(side_by_side, output_path)

    assert output_path.exists()