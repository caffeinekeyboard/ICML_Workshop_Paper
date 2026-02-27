import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from evaluation.init_gumnet import init_gumnet
from evaluation.init_fingernet import init_fingernet
from evaluation.matching_model.nbis_extractor import Nbis
from datasets.DB1_data import get_set_eval_dataloader, get_set_orig_dataloader
from datasets.kaggle_data import get_eval_dataloader, get_orig_dataloader
from model.losses.pearson_correlation_loss import PearsonCorrelationLoss
from utils.metrics import average_minutiae_match_distance
from utils.warp import warp_original


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


def plot_image_with_minutiae(image, minutiae, title="Minutiae Overlay", color="r", size=10):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(image, cmap="gray")

    xs = []
    ys = []
    for m in minutiae:
        x, y = m[0], m[1]
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)

    if xs and ys:
        ax.scatter(xs, ys, c=color, s=size)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def per_pair_average_minutiae_distance(Img1, Img2, warped_Img2 = torch.Tensor(), control_points=None, warp=True):
    if warp and control_points is not None:
        warped_Img2 = warp_original(Img2, control_points)
    
    features_Sa = extractor(Img1.cpu())
    features_Sb = extractor(Img2.cpu())

    features_warped_Sb = extractor(warped_Img2.cpu())

    plot_image_with_minutiae(Img1[0].cpu().squeeze(), features_Sa['minutiae'], title="Template Minutiae")
    plot_image_with_minutiae(Img2[0].cpu().squeeze(), features_Sb['minutiae'], title="Impression Minutiae")
    plot_image_with_minutiae(warped_Img2[0].cpu().squeeze(), features_warped_Sb['minutiae'], title="Warped Minutiae") # type: ignore
    avg_distance_before = average_minutiae_match_distance(features_Sa['minutiae'], features_Sb['minutiae'])
    avg_distance_after = average_minutiae_match_distance(features_Sa['minutiae'], features_warped_Sb['minutiae'])
    avg_distances.append(avg_distance_before)
    avg_distances_warped.append(avg_distance_after)

    print(f"Batch {idx}: Average Minutiae Distance Before Warping: {avg_distance_before:.4f}, After Warping: {avg_distance_after:.4f}")


# 1) check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2) init models
model = init_gumnet(device=device)
model.eval()

extractor = Nbis()
extractor.to(device)
extractor.eval()

# 3) init dataloaders
loader1 = get_set_eval_dataloader(
    data_root="data/FVC/FVC2004/Dbs/DB1_A",
    batch_size=16,
    num_workers=0,
)
loader2 = get_set_orig_dataloader(
    data_root="data/FVC/FVC2004/Dbs/DB1_A",
    batch_size=16,
    num_workers=0,
)
loader3 = get_eval_dataloader(
    data_root="data/Kaggle/data/5x5000/Finger_5",
    batch_size=100,
    num_workers=0,
)
loader4 = get_orig_dataloader(
    data_root="data/Kaggle/data/5x5000/Finger_5",
    batch_size=100,
    num_workers=0,
)

# 4) performance variables
avg_distances = []
avg_distances_warped = []

# 6) run inference
template_transform_eval = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(padding=(-7, -70, -8, -70), fill=255),
    transforms.Resize((192, 192)),
    transforms.RandomInvert(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
template_transform_orig = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Pad(padding=(-7, -70, -8, -70), fill=255),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

template_path = (
    "/home/marius/Asus/CVPR_Workshop/ICML_Workshop_Paper/"
    "data/Kaggle/data/5x5000/Master_Templates/5.png"
)
template_image = Image.open(template_path).convert("L")
template_tensor: torch.Tensor = template_transform_eval(template_image) # type: ignore
template_eval = template_tensor.unsqueeze(0).repeat(100, 1, 1, 1)[:100]

template_image = Image.open(template_path).convert("L")
template_tensor: torch.Tensor = template_transform_orig(template_image) # type: ignore
template_orig = template_tensor.unsqueeze(0).repeat(100, 1, 1, 1)[:100]


for idx, (eval, orig) in enumerate(zip(loader3, loader4)):
    #Sa = eval['Sa'].to(device)
    Sb = eval["Sb"].to(device)
    #orig_Sa = orig["Sa"].to(device)
    orig_Sb = orig["Sb"].to(device)
    Sa = template_eval.to(device)
    orig_Sa = template_orig.to(device)

    with torch.no_grad():
        warped_Sb, control_points = model(Sa, Sb)

    per_pair_average_minutiae_distance(orig_Sa, orig_Sb, warped_Sb, control_points=control_points, warp = True)

# 7) print results
def print_avg_minutiae_distance():
    total_before = sum(avg_distances) / len(avg_distances)
    total_after = sum(avg_distances_warped) / len(avg_distances_warped)
    percentage_improvement = ((total_before - total_after) / total_before) * 100
    print(f"Average Minutiae Distance Before Warping: {total_before:.4f}")
    print(f"Average Minutiae Distance After Warping: {total_after:.4f}")
    print(f"Percentage Improvement: {percentage_improvement:.2f}%")

print_avg_minutiae_distance()