#!/usr/bin/env python3
"""
Evaluation Script for SAM-Q
=============================

Evaluate trained models on test dataset.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/checkpoint_best.pt --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path
import json
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.data import ObjectPlacementDataModule
from src.inference import PlacementPredictor
from src.train.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM-Q model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Output results file",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config).to_dict()
    
    # Initialize data module
    data_config = config.get("data", {})
    data_module = ObjectPlacementDataModule(
        data_dir=data_config.get("root_dir", "data/"),
        batch_size=1,  # Evaluate one sample at a time
        num_workers=data_config.get("num_workers", 4),
        plane_image_size=tuple(data_config.get("plane_image_size", [1024, 1024])),
        object_image_size=tuple(data_config.get("object_image_size", [512, 512])),
    )
    data_module.setup("test")
    
    # Initialize predictor
    predictor = PlacementPredictor(checkpoint_path=args.checkpoint)
    
    # Evaluate
    test_loader = data_module.test_dataloader()
    
    all_metrics = {
        "iou": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "center_distance": [],
    }
    
    print(f"\nEvaluating on {len(test_loader)} samples...")
    
    for batch in tqdm(test_loader, desc="Testing"):
        plane_image = batch["plane_images"][0]
        object_image = batch["object_images"][0]
        text_prompt = batch["text_prompts"][0]
        target_mask = batch["masks"][0]
        
        # Predict
        results = predictor.predict(
            plane_image=plane_image,
            object_image=object_image,
            text_prompt=text_prompt,
        )
        
        # Compute metrics
        pred_mask = torch.from_numpy(results["mask"])
        metrics = compute_metrics(pred_mask, target_mask)
        
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
    
    # Average metrics
    avg_metrics = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in all_metrics.items()
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    for key, value in avg_metrics.items():
        print(f"  {key:20s}: {value:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_data = {
        "metrics": avg_metrics,
        "num_samples": len(test_loader),
        "checkpoint": args.checkpoint,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
