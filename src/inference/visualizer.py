"""
Result Visualization for SAM-Q
================================

Provides visualization tools for inference results.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_results(
    plane_image: Union[Image.Image, np.ndarray],
    results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    figsize: tuple = (15, 5),
) -> plt.Figure:
    """
    Visualize prediction results.

    Args:
        plane_image: Original plane image
        results: Prediction results from PlacementPredictor
        output_path: Path to save visualization (optional)
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Convert PIL image to numpy if needed
    if isinstance(plane_image, Image.Image):
        plane_np = np.array(plane_image)
    else:
        plane_np = plane_image

    # Panel 1: Original image + Qwen response text
    axes[0].imshow(plane_np)
    
    # Add Qwen response as text overlay
    qwen_text = results.get("qwen_response", "N/A")
    if qwen_text:
        # Truncate if too long
        if len(qwen_text) > 100:
            qwen_text = qwen_text[:97] + "..."
        textstr = f"Qwen Response:\n{qwen_text}"
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=props, color='black')
    
    axes[0].set_title("Plane/Room View", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Heatmap overlay
    axes[1].imshow(plane_np)
    heatmap = results["heatmap"]
    if hasattr(heatmap, "cpu"):
        heatmap = heatmap.float().cpu().numpy()
    # Squeeze batch and channel dimensions if present
    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()
    axes[1].imshow(heatmap, alpha=0.5)
    axes[1].set_title("Placement Probability Heatmap", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Mask with boxes
    axes[2].imshow(plane_np)
    mask = results["mask"]
    if hasattr(mask, "cpu"):
        mask = mask.float().cpu().numpy()
    # Squeeze batch and channel dimensions if present
    if mask.ndim > 2:
        mask = mask.squeeze()
    axes[2].imshow(mask, alpha=0.5, cmap="Reds")

    # Draw bounding boxes
    boxes = results["boxes"]
    scores = results["scores"]
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
        
    for i, (box, score) in enumerate(zip(boxes, scores)):
        rect = Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label=f"Box {i}: {score:.2f}",
        )
        axes[2].add_patch(rect)

    axes[2].set_title(
        f"Predicted Placement Masks ({len(results.get('boxes', []))} regions)\n"
        f"Rot: {results.get('rotation_deg', 0):.1f}° | Scale: {results.get('scale_relative', 1):.2f}x",
        fontsize=12,
        fontweight="bold"
    )
    axes[2].axis("off")
    axes[2].legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ Visualization saved to: {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def visualize_heatmap(
    plane_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    alpha: float = 0.5,
) -> plt.Figure:
    """
    Visualize only the heatmap overlay.

    Args:
        plane_image: Original plane image
        heatmap: Heatmap array
        output_path: Path to save visualization
        show: Whether to display the plot
        alpha: Heatmap transparency

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    if isinstance(plane_image, Image.Image):
        plane_np = np.array(plane_image)
    else:
        plane_np = plane_image

    ax.imshow(plane_np)
    if hasattr(heatmap, "cpu"):
        heatmap = heatmap.float().cpu().numpy()
    # Squeeze extra dimensions
    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()
    ax.imshow(heatmap, alpha=alpha)
    ax.set_title("Placement Heatmap", fontsize=16, fontweight="bold")
    ax.axis("off")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def visualize_comparison(
    plane_image: Union[Image.Image, np.ndarray],
    results_list: list,
    titles: list,
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> plt.Figure:
    """
    Compare multiple prediction results side by side.

    Args:
        plane_image: Original plane image
        results_list: List of result dictionaries
        titles: List of titles for each result
        output_path: Path to save visualization
        show: Whether to display the plot

    Returns:
        Matplotlib figure
    """
    n_results = len(results_list)
    fig, axes = plt.subplots(1, n_results, figsize=(8 * n_results, 8))
    
    if n_results == 1:
        axes = [axes]
    
    if isinstance(plane_image, Image.Image):
        plane_np = np.array(plane_image)
    else:
        plane_np = plane_image

    for idx, (results, title) in enumerate(zip(results_list, titles)):
        axes[idx].imshow(plane_np)
        heatmap = results["heatmap"]
        if hasattr(heatmap, "cpu"):
            heatmap = heatmap.float().cpu().numpy()
        # Squeeze extra dimensions
        if heatmap.ndim > 2:
            heatmap = heatmap.squeeze()
        axes[idx].imshow(heatmap, alpha=0.5)
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig
