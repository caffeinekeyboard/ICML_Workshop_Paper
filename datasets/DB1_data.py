from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SetDatasetEval(Dataset):
    """
    Dataset that loads all impressions without train/val/test split.
    Pairs each sample (Sa) with the other remaining samples (Sb).
    """
    def __init__(
        self,
        root_dir: str,
        num_images: int = 100*8,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(-200, -120, -200, -120), fill=255),
            transforms.Resize((192, 192)),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.num_images = num_images
        self.paths = []
        self.pair_indices = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        png_paths = list(self.root_dir.rglob("*.png"))
        tif_paths = list(self.root_dir.rglob("*.tif"))
        tiff_paths = list(self.root_dir.rglob("*.tiff"))
        self.paths = sorted(png_paths + tif_paths + tiff_paths, key=lambda p: str(p))
        total = len(self.paths)
        print(f"Loaded {total} samples (no split).")

        self.pair_indices = []
        for idx in range(self.num_images * 8):
            impression_idx = idx // 8
            next_set = idx // 64
            template_idx = idx - impression_idx * 8 + next_set * 8

            self.pair_indices.append((impression_idx, template_idx))

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> dict:
        template_idx, impression_idx = self.pair_indices[idx]

        img_path_Sa = self.paths[template_idx]
        img_path_Sb = self.paths[impression_idx]
        img_Sa = Image.open(img_path_Sa).convert("L")
        img_Sb = Image.open(img_path_Sb).convert("L")
        Sa_tensor = self.transform(img_Sa)
        Sb_tensor = self.transform(img_Sb)

        return {
            "Sa": Sa_tensor,
            "Sb": Sb_tensor,
            "Sa_name": img_path_Sa.name,
            "Sb_name": img_path_Sb.name,
        }

class SetDatasetOrig(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_images: int = 100*8,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(-200, -120, -200, -120), fill=255),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.num_images = num_images
        self.paths = []
        self.pair_indices = []
        self._build_dataset()

    def _build_dataset(self) -> None:
        png_paths = list(self.root_dir.rglob("*.png"))
        tif_paths = list(self.root_dir.rglob("*.tif"))
        tiff_paths = list(self.root_dir.rglob("*.tiff"))
        self.paths = sorted(png_paths + tif_paths + tiff_paths, key=lambda p: str(p))
        total = len(self.paths)
        print(f"Loaded {total} samples (no split).")

        self.pair_indices = []
        for idx in range(self.num_images * 8):
            template_idx = idx // 8
            next_set = idx // 64
            impression_idx = idx - template_idx * 8 + next_set * 8

            self.pair_indices.append((template_idx, impression_idx))

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> dict:
        template_idx, impression_idx = self.pair_indices[idx]

        img_path_Sa = self.paths[template_idx]
        img_path_Sb = self.paths[impression_idx]
        img_Sa = Image.open(img_path_Sa).convert("L")
        img_Sb = Image.open(img_path_Sb).convert("L")
        Sa_tensor = self.transform(img_Sa)
        Sb_tensor = self.transform(img_Sb)

        return {
            "Sa": Sa_tensor,
            "Sb": Sb_tensor,
            "Sa_name": img_path_Sa.name,
            "Sb_name": img_path_Sb.name,
        }


def get_set_eval_dataloader(
    data_root: str,
    batch_size: int = 8*8,
    num_workers: int = 0,
    num_images: int = 100*8,
) -> DataLoader:
    """
    Creates a single dataloader over all impressions without splitting.
    """
    dataset = SetDatasetEval(
        root_dir=data_root,
        num_images=num_images,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader

def get_set_orig_dataloader(
    data_root: str,
    batch_size: int = 8*8,
    num_workers: int = 0,
    num_images: int = 100*8,
) -> DataLoader:
    """
    Creates a single dataloader over all impressions without splitting.
    """
    dataset = SetDatasetOrig(
        root_dir=data_root,
        num_images=num_images,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader