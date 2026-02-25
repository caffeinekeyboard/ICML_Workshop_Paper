from pathlib import Path

import matplotlib.pyplot as plt
import torch

from datasets.eval_loader_data import get_dataloader


def _tensor_to_image(tensor: torch.Tensor):
    # tensor shape: (1, H, W), normalized to mean=0.5 std=0.5
    image = tensor.squeeze(0).detach().cpu() * 0.5 + 0.5
    return image.clamp(0, 1).numpy()


def main() -> None:
    data_root = (
        "/home/marius/Asus/CVPR_Workshop/ICML_Workshop_Paper/data/FCV/FVC2004/Dbs/DB1_B"
    )
    if not Path(data_root).exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    loader = get_dataloader(
        data_root=data_root,
    )

    batch = next(iter(loader))
    sa = batch["Sa"]  # (B, 1, H, W)
    sb = batch["Sb"]  # (B, 1, H, W)
    sa_names = batch["Sa_name"]
    sb_names = batch["Sb_name"]

    batch_size = sa.shape[0]

    for i in range(batch_size):
            print(f"Impression: {sb_names[i]}, Template: {sa_names[i]}")



if __name__ == "__main__":
    main()
