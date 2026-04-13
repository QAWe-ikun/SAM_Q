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
from pathlib import Path
from typing import Optional

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
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
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
        default=0.5,
        help="Confidence threshold",
    )
    predict_parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory",
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
    from src.data import ObjectPlacementDataModule
    from src.train import Trainer
    
    # Load configuration
    config = Config(args.config).to_dict()
    
    # Override with command line args
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir
    
    # Initialize data module
    data_config = config.get("data", {})
    data_module = ObjectPlacementDataModule(
        data_dir=data_config.get("root_dir", "data/"),
        batch_size=data_config.get("batch_size", 4),
        num_workers=data_config.get("num_workers", 4),
        plane_image_size=tuple(data_config.get("plane_image_size", [1024, 1024])),
        object_image_size=tuple(data_config.get("object_image_size", [512, 512])),
    )
    data_module.setup("fit")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
    )


def run_predict(args):
    """Run prediction."""
    print("=" * 60)
    print("SAM-Q Prediction")
    print("=" * 60)
    
    from src.inference import PlacementPredictor
    from PIL import Image
    import json
    from pathlib import Path
    
    # Initialize predictor
    predictor = PlacementPredictor(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    
    # Run prediction
    print(f"\nLoading images...")
    print(f"  Plane image: {args.plane_image}")
    print(f"  Object image: {args.object_image}")
    print(f"  Prompt: {args.prompt}")
    
    results = predictor.predict(
        plane_image=args.plane_image,
        object_image=args.object_image,
        text_prompt=args.prompt,
        threshold=args.threshold,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Prediction Results:")
    print(f"{'='*60}")
    print(f"  Number of detected regions: {len(results['scores'])}")
    for i, (box, score) in enumerate(zip(results['boxes'], results['scores'])):
        print(f"  Box {i}: [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}] "
              f"(score: {score:.3f})")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save visualization
    from src.inference import visualize_results
    viz_path = output_dir / "prediction.png"
    visualize_results(
        plane_image=Image.open(args.plane_image),
        results=results,
        output_path=viz_path,
    )
    
    # Save metadata
    import numpy as np
    results_json = {
        "num_placements": len(results["scores"]),
        "placements": [
            {
                "box": box.tolist(),
                "score": float(score),
            }
            for box, score in zip(results["boxes"], results["scores"])
        ],
        "image_size": list(results["image_size"]),
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
    
    from src.inference import PlacementPredictor
    from PIL import Image
    import numpy as np
    
    # Create simple demo interface
    def predict_placement(plane_img, object_img, prompt, threshold):
        """Gradio interface function."""
        # Convert numpy to PIL
        plane_pil = Image.fromarray(plane_img)
        object_pil = Image.fromarray(object_img)
        
        # Load predictor (cached)
        if not hasattr(predict_placement, "predictor"):
            predict_placement.predictor = PlacementPredictor(
                checkpoint_path="checkpoints/checkpoint_best.pt"
            )
        
        results = predict_placement.predictor.predict(
            plane_image=plane_pil,
            object_image=object_pil,
            text_prompt=prompt,
            threshold=threshold,
        )
        
        # Create visualization
        from src.inference import visualize_results
        fig = visualize_results(plane_pil, results, show=False)
        
        return fig, results["scores"].shape[0]
    
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
