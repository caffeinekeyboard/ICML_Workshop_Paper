from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SeqDatasetOrig(Dataset):
    """
    Dataset that loads all impressions without train/val/test split.
    """
    def __init__(
        self,
        root_dir: str,
        num_images: int = 10*8,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(-200, -120, -200, -120), fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.paths = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        png_paths = list(self.root_dir.rglob("*.png"))
        tif_paths = list(self.root_dir.rglob("*.tif"))
        tiff_paths = list(self.root_dir.rglob("*.tiff"))
        self.paths = sorted(png_paths + tif_paths + tiff_paths, key=lambda p: str(p))
        total = len(self.paths)
        print(f"Loaded {total} samples (no split).")

    def __getitem__(self, idx: int) -> dict:

        img_path_Sa = self.paths[idx]
        img_Sa = Image.open(img_path_Sa).convert("L")
        Sa_tensor = self.transform(img_Sa)

        return {
            "Sa": Sa_tensor,
            "Sb": Sa_tensor,
        }

    def __len__(self) -> int:
        return len(self.paths)


def get_seq_orig_dataloader(
    data_root: str,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a single dataloader over all impressions without splitting.
    """
    dataset = SeqDatasetOrig(
        root_dir=data_root,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
