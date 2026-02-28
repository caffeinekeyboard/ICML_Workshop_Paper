import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from evaluation.init_gumnet import init_gumnet
from evaluation.init_gumnet_ap import init_gumnetap
from evaluation.init_gumnet_mp import init_gumnetmp
from datasets.DB1_data import get_set_eval_dataloader # Import the specific database
from datasets.kaggle_data import get_eval_dataloader
from model.losses.pearson_correlation_loss import PearsonCorrelationLoss

"""
This script evaluates the performance of the GUMNet model on the FVC2004 dataset and the our custom Kaggle dataset.
As alignment measure we use the Pearson Correlation Coefficient (PCC) between the template and the (warped) impression.
"""

# 0) helper functions
def plot_side_by_side(image_left, image_middle, image_right, title_left="Template", title_middle="Impression", title_right="Warped"):
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    axes[0].imshow(image_left, cmap="gray")
    axes[0].set_title(title_left)
    axes[0].axis("off")

    axes[1].imshow(image_middle, cmap="gray")
    axes[1].set_title(title_middle)
    axes[1].axis("off")

    axes[2].imshow(image_right, cmap="gray")
    axes[2].set_title(title_right)
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

def per_pair_cross_correlation(Sa, Sb_CC, warped_Sb):
    before = loss(Sa, Sb_CC)
    after = loss(Sa, warped_Sb)
    cross_correlation.append(before)
    cross_correlation_warped.append(after)
    print(f"Batch {idx}: Before CC={before:.4f}, After CC={after:.4f}")


# 1) check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) init models
model = init_gumnetmp(device=device)
model.to(device)
model.eval()

# 3) init dataloaders
loader1 = get_set_eval_dataloader(
    data_root="data/FVC/FVC2004/Dbs/DB1_A",
    batch_size=64,
    num_workers=0,
    num_images=100*8, # Be aware: Set A databases contain 100*8 images, set B databases only 10*8
)
loader2 = get_eval_dataloader(
    data_root="data/Kaggle/data/5x5000/Finger_1",
    batch_size=20,
    num_workers=0,
)

# 4) performance variables
cross_correlation = []
cross_correlation_warped = []
loss = PearsonCorrelationLoss()

# 6) run inference
template_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(padding=(-7, -70, -8, -70), fill=255),
    transforms.Resize((192, 192)),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

template_path = (
    "/home/marius/Asus/CVPR_Workshop/ICML_Workshop_Paper/"
    "data/Kaggle/data/5x5000/Master_Templates/1.png"
)
template_image = Image.open(template_path).convert("L")
template_tensor: torch.Tensor = template_transform(template_image) # type: ignore
template = template_tensor.unsqueeze(0).repeat(20, 1, 1, 1)[:20] # Set this to the same batch size as the Kaggle dataloader

for idx, batch in enumerate(loader1):
    #Sa = template.to(device) # Uncomment to use the Kaggle template
    Sa = batch["Sa"].to(device)
    Sb = batch["Sb"].to(device)


    with torch.no_grad():
        warped_Sb, control_points = model(Sa, Sb)
    
    per_pair_cross_correlation(Sa, Sb, warped_Sb)

# 7) print results
def print_CC():
    total_before = sum(cross_correlation) / len(cross_correlation)
    total_after = sum(cross_correlation_warped) / len(cross_correlation_warped)
    percentage_improvement = ((total_after - total_before) / total_before) * 100
    print(f"Average Cross-Correlation Before Warping: {total_before:.4f}")
    print(f"Average Cross-Correlation After Warping: {total_after:.4f}")
    print(f"Percentage Improvement: {percentage_improvement:.2f}%")

print_CC()