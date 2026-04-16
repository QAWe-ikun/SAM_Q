#!/usr/bin/env python3
"""
SAM-Q: Main Entry Point
=========================

Unified CLI for training, inference, and evaluation of the SAM-Q system.

Usage:
    python main.py train --config configs/base.yaml
    python main.py predict --checkpoint checkpoints/best.pt --plane_image room.png --object_image chair.png
    python main.py visualize --results results.json --output output.png
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# 屏蔽无关警告
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM-Q: Intelligent Object Placement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a model
  python main.py train --config configs/base.yaml

  # Run prediction
  python main.py predict \\
    --checkpoint checkpoints/checkpoint_best.pt \\
    --plane_image examples/room.png \\
    --object_image examples/chair.png \\
    --prompt "Place the chair near the table"

  # Visualize results
  python main.py visualize --results results/output.json --output results/viz.png
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    train_parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    train_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference config (optional)",
    )
    predict_parser.add_argument(
        "--plane_image",
        type=str,
        required=True,
        help="Path to plane/room image",
    )
    predict_parser.add_argument(
        "--object_image",
        type=str,
        required=True,
        help="Path to object image",
    )
    predict_parser.add_argument(
        "--prompt",
        type=str,
        default="Place the object in a suitable position",
        help="Text prompt for placement",
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold (overrides config)",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize results")
    viz_parser.add_argument(
        "--plane_image",
        type=str,
        required=True,
        help="Path to plane image",
    )
    viz_parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results JSON file",
    )
    viz_parser.add_argument(
        "--output",
        type=str,
        default="results/visualization.png",
        help="Output image path",
    )
    viz_parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively",
    )
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Launch interactive demo")
    demo_parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Configuration file",
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate handler
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "visualize":
        run_visualize(args)
    elif args.command == "demo":
        run_demo(args)


def run_train(args):
    """Run training."""
    print("=" * 60)
    print("SAM-Q Training")
    print("=" * 60)

    from src.utils.config import Config
    from src.data.dataset import ObjectPlacementDataset
    from torch.utils.data import DataLoader
    from src.train import Trainer

    # Load configuration
    config = Config(args.config).to_dict()

    # Override with command line args
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir

    # Initialize datasets
    data_config = config.get("data", {})
    data_dir = data_config.get("root_dir", "data/")
    ann_file = data_config.get("ann_file", "annotations.json")
    seg_feature_dir = data_config.get("seg_feature_dir", None)

    train_dataset = ObjectPlacementDataset(
        data_dir=data_dir,
        split="train",
        ann_file=ann_file,
        seg_feature_dir=seg_feature_dir,
    )

    val_dataset = ObjectPlacementDataset(
        data_dir=data_dir,
        split="val",
        ann_file=ann_file,
        seg_feature_dir=seg_feature_dir,
    )

    # Create DataLoaders
    batch_size = data_config.get("batch_size", 2)
    num_workers = data_config.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset._collate_fn if hasattr(train_dataset, '_collate_fn') else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset._collate_fn if hasattr(val_dataset, '_collate_fn') else None,
    )

    print(f"\nDataset info:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data directory: {data_dir}")

    # Initialize trainer
    trainer = Trainer(config)

    # Start training
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
    )


def run_predict(args):
    """Run prediction."""
    print("=" * 60)
    print("SAM-Q Prediction")
    print("=" * 60)

    from src.models import SAMQPlacementModel
    from src.inference import visualize_results
    from src.utils.config import Config
    from PIL import Image
    import json
    import torch
    from pathlib import Path

    # Load inference config (or use defaults)
    if args.config:
        config = Config(args.config).to_dict()
    else:
        config = {"inference": {}, "model": {}}

    inference_config = config.get("inference", {})
    model_config = config.get("model", {})

    threshold = args.threshold if args.threshold is not None else inference_config.get("threshold", 0.5)
    device = inference_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output if args.output else inference_config.get("output_dir", "results/"))

    # Merge configs: checkpoint > inference config > defaults
    qwen_config = config.get("qwen", model_config.get("qwen", {}))
    sam3_config = config.get("sam3", model_config.get("sam3", {}))
    adapter_config = config.get("adapter", model_config.get("adapter", {}))
    action_head_config = config.get("action_head", model_config.get("action_head", {}))

    # Initialize model
    model = SAMQPlacementModel(
        sam3_checkpoint_path=sam3_config.get("checkpoint_path"),
        qwen_model_name=qwen_config.get("model_name"),
        qwen_lora_path=qwen_config.get("lora_path"),
        sam3_input_dim=sam3_config.get("input_dim", 256),
        qwen_hidden_dim=qwen_config.get("hidden_dim", 4096),
        adapter_hidden_dim=adapter_config.get("hidden_dim", 512),
        num_seg_tokens=config.get("num_seg_tokens", 1),
        device=device,
        action_head_config=action_head_config,
    )

    # Load trained weights
    model.load_all(eval_mode=True)  # Initialize Qwen + SAM3 structures

    # Determine checkpoint loading strategy
    adapter_ckpt = inference_config.get("adapter_checkpoint_path")
    sam3_ckpt = inference_config.get("sam3_checkpoint_path")
    full_ckpt = inference_config.get("checkpoint_path", args.checkpoint)

    if adapter_ckpt and sam3_ckpt:
        # Option 2: Load split checkpoints
        model.load_split_checkpoint(
            adapter_checkpoint_path=adapter_ckpt,
            sam3_checkpoint_path=sam3_ckpt,
        )
    else:
        # Option 1: Load combined checkpoint
        checkpoint = torch.load(full_ckpt, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        fixed_state_dict = {}
        sam3_prefixes = ["backbone.", "geometry_encoder.", "transformer.", "segmentation_head.", "dot_prod_scoring."]
        
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in sam3_prefixes):
                fixed_state_dict[f"sam3_loader.model.{k}"] = v
            else:
                fixed_state_dict[k] = v
                
        result = model.load_state_dict(fixed_state_dict, strict=False)
        print(f"Loaded full checkpoint: {full_ckpt}")
        if result.missing_keys or result.unexpected_keys:
            print(f"  Note: Some keys did not match (Expected for older checkpoints).")

    # Load images
    plane_image = Image.open(args.plane_image).convert("RGB")
    object_image = Image.open(args.object_image).convert("RGB")

    # Run prediction
    print(f"\nLoading images...")
    print(f"  Plane image: {args.plane_image}")
    print(f"  Object image: {args.object_image}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Threshold: {threshold}")

    output = model.predict(
        plane_image=plane_image,
        text_prompt=args.prompt,
        images=[plane_image, object_image],
        threshold=threshold,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"Prediction Results:")
    print(f"{'='*60}")
    print(f"  Heatmap shape: {output['heatmap'].shape}")
    print(f"  Best candidate: #{output.get('best_candidate_idx', 0)}")
    print(f"  Rotation 6D: {output['rotation_6d'][0].tolist()}")
    print(f"  Scale: {output['scale_relative'][0].item():.3f}")
    if output.get("qwen_response"):
        print(f"  Qwen Response: {output['qwen_response'][:100]}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save visualization
    if inference_config.get("save_visualizations", True):
        viz_path = output_dir / "prediction.png"
        visualize_results(
            plane_image=plane_image,
            results={
                "heatmap": output["heatmap"],
                "mask": output["binary_heatmap"],
                "rotation_deg": output.get("rotation_deg", 0),
                "scale_relative": output.get("scale_relative", 1.0),
                "qwen_response": output.get("qwen_response", ""),
            },
            output_path=viz_path,
        )

    # Save metadata
    if inference_config.get("save_json", True):
        results_json = {
            "rotation_6d": output["rotation_6d"].tolist(),
            "rotation_matrix": output["rotation_matrix"].tolist(),
            "scale_relative": output["scale_relative"].tolist(),
            "heatmap_shape": list(output["heatmap"].shape),
        }

        json_path = output_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2)

        print(f"\n✓ Results saved to:")
        print(f"  Visualization: {viz_path}")
        print(f"  Metadata: {json_path}")


def run_visualize(args):
    """Run visualization."""
    print("=" * 60)
    print("SAM-Q Visualization")
    print("=" * 60)
    
    import json
    from src.inference import visualize_results
    from PIL import Image
    
    # Load results
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Load plane image
    plane_image = Image.open(args.plane_image)
    
    # Visualize
    visualize_results(
        plane_image=plane_image,
        results=results,
        output_path=args.output,
        show=args.show,
    )


def run_demo(args):
    """Run interactive demo."""
    print("=" * 60)
    print("SAM-Q Interactive Demo")
    print("=" * 60)
    print("\nNote: Gradio is required for the demo.")
    print("Install with: pip install gradio")

    try:
        import gradio as gr
    except ImportError:
        print("\nError: Gradio not installed.")
        print("Install with: pip install gradio")
        return

    from src.models import SAMQPlacementModel
    from src.inference import visualize_results
    from PIL import Image
    import numpy as np
    import torch

    # Load model (cached)
    def get_model():
        if not hasattr(get_model, "_model"):
            checkpoint = torch.load("checkpoints/checkpoint_best.pt", map_location="cpu")
            model_config = checkpoint.get("config", {}).get("model", {})

            get_model._model = SAMQPlacementModel(
                qwen_model_name=model_config.get("qwen", {}).get(
                    "model_name", "./models/qwen3_vl"
                ),
                sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
                qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
                adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
                num_seg_tokens=model_config.get("num_seg_tokens", 1),
                device="cuda" if torch.cuda.is_available() else "cpu",
                action_head_config=model_config.get("action_head", {"heatmap_size": 64}),
            )
            get_model._model.load_state_dict(checkpoint["model_state_dict"])
            get_model._model.eval()
        return get_model._model

    # Create simple demo interface
    def predict_placement(plane_img, object_img, prompt, threshold):
        """Gradio interface function."""
        # Convert numpy to PIL
        plane_pil = Image.fromarray(plane_img)
        object_pil = Image.fromarray(object_img)

        model = get_model()

        output = model.predict(
            plane_image=plane_pil,
            text_prompt=prompt,
            images=[plane_pil, object_pil],
            threshold=threshold,
        )

        # Create visualization
        fig = visualize_results(
            plane_pil,
            {"heatmap": output["heatmap"], "binary_heatmap": output["binary_heatmap"]},
            show=False,
        )

        return fig, output["rotation_6d"].tolist()
    
    # Build Gradio interface
    with gr.Blocks(title="SAM-Q Demo") as demo:
        gr.Markdown("# SAM-Q: Intelligent Object Placement")
        gr.Markdown("Upload a room image and object image to predict optimal placement.")
        
        with gr.Row():
            with gr.Column():
                plane_input = gr.Image(label="Room Image")
                object_input = gr.Image(label="Object Image")
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Place the object in a suitable position",
                )
                threshold_input = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                )
                submit_btn = gr.Button("Predict", variant="primary")
            
            with gr.Column():
                output_plot = gr.Plot(label="Prediction Results")
                count_output = gr.Number(label="Number of Placements")
        
        submit_btn.click(
            fn=predict_placement,
            inputs=[plane_input, object_input, prompt_input, threshold_input],
            outputs=[output_plot, count_output],
        )
    
    demo.launch()


if __name__ == "__main__":
    main()
