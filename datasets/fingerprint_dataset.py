from pathlib import Path
from PIL import Image
from typing import Optional, List, Callable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class FingerprintGumNetDataset(Dataset):
    """
    PyTorch Dataset for 2D GumNet Siamese Training.
    Pairs a noisy impression (Sb) with its uniquely corresponding master template (Sa) found in the pattern's 'master' directory.
    """
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train', 
                 noise_levels: Optional[List[str]] = None,
                 impression_transform: Optional[Callable] = None,
                 affine_degrees: int = 3,
                 affine_translate: float = 0.05):
        """
        Args:
            root_dir (str): Path to the 'data' directory.
            split (str): One of 'train', 'val', or 'test'.
            noise_levels (list): Specific noise levels to include (e.g., ['Noise_Level_10']). 
                                 If None, uses all available noise levels.
            impression_transform (callable): Transformations applied to Sb.
            affine_degrees (int): Max degrees for random rotation in the affine transform.
            affine_translate (float): Max translation as a fraction of image size for the affine transform.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.template_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((192, 192)),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
        ])
        self.impression_transform = impression_transform if impression_transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((192, 192)),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
        ])
        self.pairs = []
        self._build_dataset(noise_levels)

    def _build_dataset(self, noise_levels: Optional[List[str]]):
        pattern_classes = [d for d in self.root_dir.iterdir() if d.is_dir()]

        for pattern_dir in pattern_classes:
            master_dir = pattern_dir / 'master'

            try:
                master_path = next(master_dir.glob('*.png'))
            except StopIteration:
                print(f"Warning: No master template found in {master_dir}. Skipping pattern.")
                continue

            subdirs = [d for d in pattern_dir.iterdir() if d.is_dir() and d.name.startswith('Noise_Level')]
            if noise_levels:
                subdirs = [d for d in subdirs if d.name in noise_levels]

            for noise_dir in subdirs:
                split_dir = noise_dir / self.split
                if not split_dir.exists():
                    continue

                for imp_path in split_dir.glob('*.png'):
                    self.pairs.append({
                        'Sa_path': str(master_path),
                        'Sb_path': str(imp_path),
                        'pattern': pattern_dir.name,
                        'noise_level': noise_dir.name
                    })

        print(f"Loaded {len(self.pairs)} {self.split} pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair_info = self.pairs[idx]
        Sa_img = Image.open(pair_info['Sa_path']).convert('L')
        Sb_img = Image.open(pair_info['Sb_path']).convert('L')
        Sa_tensor = self.template_transform(Sa_img)
        Sb_tensor = self.impression_transform(Sb_img)
            
        return {
            'Sa': Sa_tensor,
            'Sb': Sb_tensor,
            'pattern': pair_info['pattern'],
            'noise_level': pair_info['noise_level']
        }




def get_dataloaders(data_root: str, batch_size: int = 16, num_workers: int = 4, noise_levels: Optional[List[str]] = None):
    """
    Instantiates the Train, Val, and Test dataloaders.
    """
    train_dataset = FingerprintGumNetDataset(root_dir=data_root, split='train', noise_levels=noise_levels)
    val_dataset = FingerprintGumNetDataset(root_dir=data_root, split='val', noise_levels=noise_levels)
    test_dataset = FingerprintGumNetDataset(root_dir=data_root, split='test', noise_levels=noise_levels)

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True,
                              drop_last=True)
                              
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
                            
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
                             
    return train_loader, val_loader, test_loader