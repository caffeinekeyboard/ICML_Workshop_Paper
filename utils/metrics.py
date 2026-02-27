import torch
from typing import List


def average_minutiae_match_distance(
    minutiae_list_1: List[torch.Tensor],
    minutiae_list_2: List[torch.Tensor],
):
    """
    Compute average matched distance per pair of minutiae sets using greedy closest matching.

    For each pair (minutiae_list_1[i], minutiae_list_2[i]), repeatedly match the closest
    remaining minutiae between the two sets until one set is exhausted. The result is the
    average distance of all matched pairs for each batch item.

    Args:
        minutiae_list_1: List of minutiae tensors, shape (N_i, >=2)
        minutiae_list_2: List of minutiae tensors, shape (M_i, >=2)

    Returns:
        torch.Tensor of shape (batch_size,) with average distances. If no matches, 0.0.
    """
    batch_size = min(len(minutiae_list_1), len(minutiae_list_2))
    device = minutiae_list_1[0].device if batch_size > 0 else "cpu"
    avg_distances = torch.zeros(batch_size, device=device, dtype=torch.float32)
    avg_distance = 0.0

    for i in range(batch_size):
        m1 = minutiae_list_1[i]
        m2 = minutiae_list_2[i]
        if m1.numel() == 0 or m2.numel() == 0:
            avg_distances[i] = torch.tensor(0.0, device=device)
            continue

        coords_1 = m1[:, :2]
        coords_2 = m2[:, :2]

        distances = torch.cdist(coords_1, coords_2, p=2)
        num_matches = min(coords_1.shape[0], coords_2.shape[0])
        matched_dists = []

        distances_work = distances.clone()
        for _ in range(num_matches):
            min_val, min_idx = torch.min(distances_work.view(-1), dim=0)
            if torch.isinf(min_val):
                break
            row = min_idx // distances_work.shape[1]
            col = min_idx % distances_work.shape[1]
            matched_dists.append(min_val)

            distances_work[row, :] = torch.inf
            distances_work[:, col] = torch.inf

        if len(matched_dists) > 0:
            avg_distances[i] = torch.stack(matched_dists).mean().to(torch.float32)
        else:
            avg_distances[i] = torch.tensor(0.0, device=device)
    
    avg_distance = avg_distances.mean().item() if batch_size > 0 else 0.0

    return avg_distance