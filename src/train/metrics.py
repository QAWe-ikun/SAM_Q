"""
Evaluation Metrics for SAM-Q
==============================

Provides metrics computation for object placement evaluation.
"""

import torch # type: ignore
import numpy as np
from typing import Dict, Tuple


def compute_iou(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute Intersection over Union (IoU).
    
    Args:
        pred_mask: Predicted mask
        target_mask: Ground truth mask
        threshold: Binarization threshold
        
    Returns:
        IoU score
    """
    pred_binary = (pred_mask > threshold).float()
    target_binary = (target_mask > 0.5).float()
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def compute_precision_at_k(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    k: float = 0.5,
) -> float:
    """
    Compute precision at threshold k.
    
    Args:
        pred_mask: Predicted mask
        target_mask: Ground truth mask
        k: Threshold
        
    Returns:
        Precision score
    """
    pred_binary = (pred_mask > k).float()
    target_binary = (target_mask > 0.5).float()
    
    true_positives = (pred_binary * target_binary).sum()
    predicted_positives = pred_binary.sum()
    
    if predicted_positives == 0:
        return 0.0
    
    return (true_positives / predicted_positives).item()


def compute_recall_at_k(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    k: float = 0.5,
) -> float:
    """
    Compute recall at threshold k.
    
    Args:
        pred_mask: Predicted mask
        target_mask: Ground truth mask
        k: Threshold
        
    Returns:
        Recall score
    """
    pred_binary = (pred_mask > k).float()
    target_binary = (target_mask > 0.5).float()
    
    true_positives = (pred_binary * target_binary).sum()
    actual_positives = target_binary.sum()
    
    if actual_positives == 0:
        return 0.0
    
    return (true_positives / actual_positives).item()


def compute_f1_score(
    precision: float,
    recall: float,
    epsilon: float = 1e-7,
) -> float:
    """
    Compute F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        epsilon: Small constant for numerical stability
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall + epsilon)


def compute_center_distance(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    """
    Compute Euclidean distance between mask centers.
    
    Args:
        pred_mask: Predicted mask
        target_mask: Ground truth mask
        
    Returns:
        Center distance in pixels
    """
    def get_center(mask: torch.Tensor) -> Tuple[float, float]:
        coords = torch.nonzero(mask > 0.5)
        if len(coords) == 0:
            return (0.0, 0.0)
        return (coords[:, 0].float().mean().item(), 
                coords[:, 1].float().mean().item())
    
    pred_center = get_center(pred_mask)
    target_center = get_center(target_mask)
    
    distance = np.sqrt(
        (pred_center[0] - target_center[0]) ** 2 +
        (pred_center[1] - target_center[1]) ** 2
    )
    
    return distance


def compute_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        pred_mask: Predicted mask
        target_mask: Ground truth mask
        threshold: Binarization threshold
        
    Returns:
        Dictionary of metrics
    """
    iou = compute_iou(pred_mask, target_mask, threshold)
    precision = compute_precision_at_k(pred_mask, target_mask, threshold)
    recall = compute_recall_at_k(pred_mask, target_mask, threshold)
    f1 = compute_f1_score(precision, recall)
    center_dist = compute_center_distance(pred_mask, target_mask)
    
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "center_distance": center_dist,
    }
