"""
VLA Inference Script
"""

import argparse
import sys
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sam2qhmvpl_system import SAM2QVLAIncremental


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--plane_image", type=str, required=True)
    parser.add_argument("--object_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the new incremental VLA model
    model = SAM2QVLAIncremental()
    
    # Load checkpoint - this might need adjustment depending on saved format
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    except Exception as e:
        print(f"Warning: Could not load checkpoint properly: {e}")
        print("Initializing with random weights...")
    
    model.to(device)
    model.eval()

    # Load images
    scene_image = Image.open(args.plane_image).convert("RGB")
    object_image = Image.open(args.object_image).convert("RGB")

    # Preprocess images
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to standard size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    scene_tensor = preprocess(scene_image).unsqueeze(0).to(device)  # Add batch dimension
    object_tensor = preprocess(object_image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Add alpha channel to object tensor if needed (RGBA format)
    if object_tensor.shape[1] == 3:
        alpha_channel = torch.ones(1, 1, object_tensor.shape[2], object_tensor.shape[3]).to(device)
        object_tensor = torch.cat([object_tensor, alpha_channel], dim=1)

    text_query = [args.prompt]

    with torch.no_grad():
        output = model(
            object_img=object_tensor,
            scene_img=scene_tensor,
            text_query=text_query,
            return_intermediate=True
        )

    print(f"Predicted pose: {output['best_pose'][0].cpu().numpy()}")
    print(f"Confidence score: {output['best_score'][0].cpu().item():.3f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original scene
    axes[0].imshow(scene_image)
    axes[0].set_title("Scene")
    axes[0].axis('off')
    
    # Object
    axes[1].imshow(object_image)
    axes[1].set_title("Object to Place")
    axes[1].axis('off')
    
    # Show placement visualization (simplified)
    # Since we don't have masks in the new system, we'll visualize the grounding map
    if 'intermediate_results' in output:
        grounding_map = output['intermediate_results'].get('qwen_output', {}).get('grounding_maps')
        if grounding_map is not None:
            axes[2].imshow(grounding_map[0, 0].cpu().numpy(), cmap="hot", alpha=0.6)
    axes[2].imshow(scene_image)
    axes[2].set_title("Suggested Placement Area")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    import torchvision.transforms as transforms
    main()
