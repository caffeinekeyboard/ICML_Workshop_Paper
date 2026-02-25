from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FingerprintGumNetNoSplitDataset(Dataset):
    """
    Dataset that loads all impressions without train/val/test split.
    Pairs each impression (Sb) with its master template (Sa).
    """
    def __init__(
        self,
        root_dir: str,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(80, 0, 80, 0), fill=255),
            transforms.Resize((192, 192)),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.paths = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        png_paths = list(self.root_dir.rglob("*.png"))
        tif_paths = list(self.root_dir.rglob("*.tif"))
        tiff_paths = list(self.root_dir.rglob("*.tiff"))
        self.paths = png_paths + tif_paths + tiff_paths
        print(f"Loaded {len(self.paths)} samples (no split).")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("L")
        Sa_tensor = self.transform(img)
        Sb_tensor = self.transform(img)

        return {
            "Sa": Sa_tensor,
            "Sb": Sb_tensor,
            "path": str(img_path),
        }


def get_no_split_dataloader(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """
    Creates a single dataloader over all impressions without splitting.
    """
    dataset = FingerprintGumNetNoSplitDataset(
        root_dir=data_root,
    )

    has_samples = len(dataset) > 0
    if not has_samples:
        print(
            "Warning: no samples found for no-split dataloader. "
            "Check the data_root path and that it contains .png/.tif/.tiff files."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=has_samples,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=has_samples,
    )

    return loader
