import torch
from typing import List


def minutiae_metric(
    minutiae_list_1: List[torch.Tensor],
    minutiae_list_2: List[torch.Tensor],
    distance_threshold: float = 6.0,
    angle_threshold: float = 0.2,
    match_ratio_threshold: float = 0.0,
) -> torch.Tensor:
    """
    Compute pairwise fingerprint matching scores from minutiae (assumes aligned).
    
    Returns:
        torch.Tensor of shape (batch_size_1, batch_size_2) with scores in [0, 1]
    """
    batch_size_1, batch_size_2 = len(minutiae_list_1), len(minutiae_list_2)
    device = minutiae_list_1[0].device if batch_size_1 > 0 else "cpu"
    score_matrix = torch.zeros(batch_size_1, batch_size_2, device=device)
    
    for i in range(batch_size_1):
        for j in range(batch_size_2):
            m1, m2 = minutiae_list_1[i], minutiae_list_2[j]
            if m1.shape[0] > 0 and m2.shape[0] > 0:
                score_matrix[i, j] = _compute_match_score(
                    m1, m2, distance_threshold, angle_threshold, match_ratio_threshold
                )
    
    return score_matrix


def _compute_match_score(
    m1: torch.Tensor,
    m2: torch.Tensor,
    dist_thresh: float,
    angle_thresh: float,
    ratio_thresh: float,
) -> torch.Tensor:
    """Compute matching score between two minutiae sets."""
    match_count = _count_matches(m1[:, :2], m1[:, 2], m2[:, :2], m2[:, 2], dist_thresh, angle_thresh)
    match_ratio = (2.0 * match_count) / (m1.shape[0] + m2.shape[0])
    score = 0.0 if match_ratio < ratio_thresh else min(1.0, match_ratio)
    return torch.tensor(score, device=m1.device, dtype=torch.float32)


def _count_matches(
    coords_1: torch.Tensor,
    angles_1: torch.Tensor,
    coords_2: torch.Tensor,
    angles_2: torch.Tensor,
    dist_thresh: float,
    angle_thresh: float,
) -> int:
    """Count matching minutiae pairs via greedy matching."""
    distances = torch.cdist(coords_1, coords_2, p=2)
    angle_diffs = torch.abs(angles_1.unsqueeze(1) - angles_2.unsqueeze(0))
    angle_diffs = torch.remainder(angle_diffs + torch.pi, 2 * torch.pi) - torch.pi
    angle_diffs = torch.abs(angle_diffs)
    
    valid_matches = (distances <= dist_thresh) & (angle_diffs <= angle_thresh)
    matched = set()
    match_count = 0
    
    for i in range(coords_1.shape[0]):
        valid_idx = torch.where(valid_matches[i])[0]
        candidates = valid_idx[[j.item() not in matched for j in valid_idx]]
        if len(candidates) > 0:
            closest = candidates[torch.argmin(distances[i, candidates])].item()
            matched.add(closest)
            match_count += 1
    
    return match_count


def orientation_metric(
    orientation_list_1: List[torch.Tensor],
    orientation_list_2: List[torch.Tensor],
    mask_list_1: List[torch.Tensor] | None = None,
    mask_list_2: List[torch.Tensor] | None = None,
    angle_threshold: float = torch.pi / 6,
) -> torch.Tensor:
    """
    Compute pairwise fingerprint matching scores from orientation fields (assumes aligned).
    
    Args:
        orientation_list_1: List of orientation field tensors of shape (H, W) in radians
        orientation_list_2: List of orientation field tensors of shape (H, W) in radians
        mask_list_1: Optional list of segmentation masks of shape (H, W) for orientation_list_1
        mask_list_2: Optional list of segmentation masks of shape (H, W) for orientation_list_2
        angle_threshold: Threshold for angle difference in radians to consider match
    
    Returns:
        torch.Tensor of shape (batch_size_1, batch_size_2) with scores in [0, 1]
    """
    batch_size_1, batch_size_2 = len(orientation_list_1), len(orientation_list_2)
    device = orientation_list_1[0].device if batch_size_1 > 0 else "cpu"
    score_matrix = torch.zeros(batch_size_1, batch_size_2, device=device)
    
    for i in range(batch_size_1):
        for j in range(batch_size_2):
            ori_1 = orientation_list_1[i]
            ori_2 = orientation_list_2[j]
            mask_1 = mask_list_1[i] if mask_list_1 is not None else None
            mask_2 = mask_list_2[j] if mask_list_2 is not None else None
            
            score_matrix[i, j] = _compute_orientation_similarity(
                ori_1, ori_2, mask_1, mask_2, angle_threshold
            )
    
    return score_matrix


def _compute_orientation_similarity(
    ori_1: torch.Tensor,
    ori_2: torch.Tensor,
    mask_1: torch.Tensor | None = None,
    mask_2: torch.Tensor | None = None,
    angle_threshold: float = torch.pi / 6,
) -> torch.Tensor:
    """
    Compute orientation field similarity score between two orientation maps.
    
    Uses circular distance and masking to compute a similarity metric in [0, 1].
    """
    # Ensure tensors are on the same device
    device = ori_1.device
    ori_2 = ori_2.to(device)
    
    # Pad/crop to same size (use minimum size)
    h1, w1 = ori_1.shape
    h2, w2 = ori_2.shape
    h_min, w_min = min(h1, h2), min(w1, w2)
    
    ori_1_cropped = ori_1[:h_min, :w_min]
    ori_2_cropped = ori_2[:h_min, :w_min]
    
    # Handle masks
    if mask_1 is not None:
        mask_1 = mask_1[:h_min, :w_min].to(device)
    if mask_2 is not None:
        mask_2 = mask_2[:h_min, :w_min].to(device)
    
    # Compute circular angle difference
    angle_diff = torch.abs(ori_1_cropped - ori_2_cropped)
    angle_diff = torch.minimum(angle_diff, 2 * torch.pi - angle_diff)
    
    # Compute similarity: 1 where angles are close, 0 where far
    angle_similarity = torch.exp(-angle_diff / (2 * angle_threshold ** 2))
    
    # Apply masks if provided
    if mask_1 is not None and mask_2 is not None:
        # Only compare pixels where both masks are active
        combined_mask = (mask_1 > 0) & (mask_2 > 0)
    elif mask_1 is not None:
        combined_mask = mask_1 > 0
    elif mask_2 is not None:
        combined_mask = mask_2 > 0
    else:
        combined_mask = torch.ones_like(ori_1_cropped, dtype=torch.bool)
    
    # Compute mean similarity over valid pixels
    if combined_mask.sum() > 0:
        score = angle_similarity[combined_mask].mean()
    else:
        score = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    return score.to(torch.float32)


def cosine_similarity_metric(
    image_list_1: List[torch.Tensor],
    image_list_2: List[torch.Tensor],
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between flattened image representations.
    
    Args:
        image_list_1: List of image tensors of shape (...) to be flattened
        image_list_2: List of image tensors of shape (...) to be flattened
    
    Returns:
        torch.Tensor of shape (batch_size_1, batch_size_2) with cosine similarity scores in [-1, 1]
    """
    batch_size_1, batch_size_2 = len(image_list_1), len(image_list_2)
    device = image_list_1[0].device if batch_size_1 > 0 else "cpu"
    score_matrix = torch.zeros(batch_size_1, batch_size_2, device=device)
    
    # Flatten all images
    flattened_1 = torch.stack([img.flatten() for img in image_list_1])
    flattened_2 = torch.stack([img.flatten() for img in image_list_2])
    
    # Normalize for cosine similarity
    flattened_1_norm = torch.nn.functional.normalize(flattened_1, p=2, dim=1)
    flattened_2_norm = torch.nn.functional.normalize(flattened_2, p=2, dim=1)
    
    # Compute pairwise cosine similarity via dot product of normalized vectors
    score_matrix = torch.mm(flattened_1_norm, flattened_2_norm.t())
    
    return score_matrix

def hybrid_metric(warped_impression, reference_image, warped_features, reference_features, alpha: float = 0.5, beta: float = 0.5, delta: float = 0.2, mask: bool = False):
    """
    Compute a hybrid matching score that combines minutiae, orientation and cosine similarity metrics.
    
    This is a simple weighted average of the three metrics, but could be extended to more complex combinations.
    """
    if mask:
        minutiae_score = minutiae_metric(warped_features['minutiae'], reference_features['minutiae'])
        orientation_score = orientation_metric(warped_features['orientation_field'], reference_features['orientation_field'], mask_list_1=warped_features['segmentation_mask'], mask_list_2=reference_features['segmentation_mask'])
    else:
        minutiae_score = minutiae_metric(warped_features['minutiae'], reference_features['minutiae'])
        orientation_score = orientation_metric(warped_features['orientation_field'], reference_features['orientation_field'])
    
    cosine_score = cosine_similarity_metric([warped_impression], [reference_image])
    
    combined_score = alpha * minutiae_score + beta * orientation_score + delta * cosine_score
    return combined_score