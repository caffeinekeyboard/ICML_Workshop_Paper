"""Matching performance metrics for biometric systems."""

import torch
from typing import Tuple
import matplotlib.pyplot as plt


def compute_far_frr(score_matrix: torch.Tensor, threshold: float) -> Tuple[float, float]:
    """Compute FAR and FRR for a given threshold.
    
    Args:
        score_matrix: NxN score matrix (diagonal=genuine, off-diagonal=impostor)
        threshold: Decision threshold (scores >= threshold are accepted)
    
    Returns:
        (FAR, FRR) where FAR is impostor acceptance rate, FRR is genuine rejection rate
    """
    n = score_matrix.shape[0]
    genuine_scores = torch.diag(score_matrix)
    impostor_scores = score_matrix[~torch.eye(n, dtype=torch.bool, device=score_matrix.device)]
    
    frr = (genuine_scores < threshold).float().mean().item()
    far = (impostor_scores >= threshold).float().mean().item()
    
    return far, frr

def fcv_frr(score_matrix: torch.Tensor, threshold: float = 0.54) -> torch.Tensor:
    """Compute the FCV competition specific FRR for a given threshold.
    
    Args:
        score_matrix: NxN score matrix (diagonal=genuine, off-diagonal=impostor)
        threshold: Decision threshold (scores >= threshold are accepted)
    
    Returns:
        FRR, where FRR is the genuine rejection rate
    """
    n, m = score_matrix.shape[0], score_matrix.shape[1]
    mask = design_mask(n,m)
    genuine_scores = score_matrix[mask]
    
    frr = (genuine_scores < threshold).float().mean()
    
    return frr


def design_mask(n: int, m: int) -> torch.Tensor:
    """Create a mask for the FCV competition to exclude certain pairs from evaluation."""
    mask = torch.ones((n, m), dtype=torch.bool)
    for i in range(n):
        for j in range(m):
            if i == j:  # Exclude pairs from the same subject
                mask[i, j] = False
            if j<i:  # Exclude pairs where j < i to avoid double counting
                mask[i, j] = False
            if (((i // 8) + 1) * 8) <= j: # Exclude pairs of different fingerprints
                mask[i, j] = False
    return mask


def compute_far_frr_curve(
    score_matrix: torch.Tensor, num_thresholds: int = 1000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FAR and FRR curves across thresholds.
    
    Args:
        score_matrix: NxN score matrix (diagonal=genuine, off-diagonal=impostor)
        num_thresholds: Number of threshold values to evaluate
    
    Returns:
        (thresholds, far_values, frr_values)
    """
    n = score_matrix.shape[0]
    device = score_matrix.device
    genuine_scores = torch.diag(score_matrix)
    impostor_scores = score_matrix[~torch.eye(n, dtype=torch.bool, device=device)]
    
    thresholds = torch.linspace(score_matrix.min().item(), score_matrix.max().item(), 
                                num_thresholds, device=device)
    
    # Vectorized computation
    frr_values = (genuine_scores[:, None] < thresholds).float().mean(dim=0)
    far_values = (impostor_scores[:, None] >= thresholds).float().mean(dim=0)
    
    return thresholds, far_values, frr_values


def find_eer_threshold(
    score_matrix: torch.Tensor, num_thresholds: int = 1000
) -> Tuple[float, float, int]:
    """Find Equal Error Rate (EER) where FAR equals FRR.
    
    Args:
        score_matrix: NxN score matrix (diagonal=genuine, off-diagonal=impostor)
        num_thresholds: Number of threshold values to search
    
    Returns:
        (eer, threshold, eer_index)
    """
    thresholds, far_values, frr_values = compute_far_frr_curve(score_matrix, num_thresholds)
    
    eer_index = torch.argmin(torch.abs(far_values - frr_values)).item()
    eer = ((far_values[eer_index] + frr_values[eer_index]) / 2.0).item() # type: ignore
    
    return eer, thresholds[eer_index].item(), eer_index # type: ignore


def compute_rank_n_accuracy(score_matrix: torch.Tensor, n: int = 1) -> float:
    """Compute Rank-N identification accuracy.
    
    Args:
        score_matrix: NxN score matrix (diagonal contains genuine matches)
        n: Rank to evaluate (1 for top-1 accuracy, 5 for top-5, etc.)
    
    Returns:
        Rank-N accuracy (proportion where true match is in top-N)
    """
    num_samples = score_matrix.shape[0]
    device = score_matrix.device
    top_n_indices = torch.argsort(score_matrix, dim=1)[:, -n:]
    correct = torch.any(top_n_indices == torch.arange(num_samples, device=device)[:, None], dim=1)
    
    return correct.float().mean().item()


def compute_genuine_impostor_distributions(
    score_matrix: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract genuine and impostor score distributions.
    
    Args:
        score_matrix: NxN score matrix (diagonal=genuine, off-diagonal=impostor)
    
    Returns:
        (genuine_scores, impostor_scores)
    """
    n = score_matrix.shape[0]
    genuine_scores = torch.diag(score_matrix)
    impostor_scores = score_matrix[~torch.eye(n, dtype=torch.bool, device=score_matrix.device)]
    
    return genuine_scores, impostor_scores
