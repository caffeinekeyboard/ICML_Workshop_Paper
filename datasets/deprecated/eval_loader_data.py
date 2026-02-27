from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class EvalDataset(Dataset):
    """
    Dataset that loads all impressions without train/val/test split.
    Pairs each sample (Sb) with the other remaining samples (Sa).
    """
    def __init__(
        self,
        root_dir: str,
        num_sets: int = 10,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad(padding=(0, 80, 0, 80), fill=255),
            transforms.Resize((192, 192)),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.samples_per_set = 8
        self.num_sets = num_sets
        self.idx_per_set = (self.samples_per_set * (self.samples_per_set - 1)) // 2
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

        if total < self.samples_per_set:
            print(
                "Warning: not enough samples to build a single set. "
                f"Found {total}, expected at least {self.samples_per_set}."
            )
            self.num_sets = 0
            self.pair_indices = []
            return

        available_sets = total // self.samples_per_set
        if total % self.samples_per_set != 0:
            print(
                "Warning: sample count is not divisible by samples_per_set. "
                f"Truncating to {available_sets * self.samples_per_set} samples."
            )

        if self.num_sets > available_sets:
            print(
                f"Warning: requested num_sets={self.num_sets} exceeds available_sets={available_sets}. "
                "Truncating num_sets."
            )
            self.num_sets = available_sets

        self.pair_indices = []
        for idx in range(self.num_sets * self.idx_per_set):
            current_set = idx // self.idx_per_set
            if idx == current_set * self.idx_per_set:
                self.next_set = True
            if self.next_set:
                self.running_id = 1
                self.running_max = 8
                self.next_set = False
            if self.running_id == self.running_max:
                self.running_id = 1
                self.running_max -= 1
            diff = 8 - self.running_max
            template_idx = (diff + self.running_id + current_set * 8)
            self.running_id += 1

            impression_idx = current_set * 8 + 8 - self.running_max

            self.pair_indices.append((impression_idx, template_idx))

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> dict:
        impression_idx, template_idx = self.pair_indices[idx]

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


def get_eval_dataloader(
    data_root: str,
    batch_size: int = (8*7) // 2, ### NEVER CHANGE THAT ###
    num_workers: int = 0, ### NEVER CHANGE THAT ###
) -> DataLoader:
    """
    Creates a single dataloader over all impressions without splitting.
    """
    dataset = EvalDataset(
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
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader
