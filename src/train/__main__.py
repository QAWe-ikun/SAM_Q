#!/usr/bin/env python3
"""
Training Script for SAM-Q
==========================

Standalone training entry point.

Usage:
    python -m src.train --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import Config
from src.data import ObjectPlacementDataModule
from src.train import Trainer


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train SAM-Q object placement model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config).to_dict()
    
    # Override with command line args
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    
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


if __name__ == "__main__":
    main()
