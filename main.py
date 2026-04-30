#!/usr/bin/env python3
"""
SAM-Q: Main Entry Point
=========================

Unified CLI for training, inference, and evaluation of the SAM-Q system.

Usage:
    python main.py train --config configs/stage1_qwen_lora.yaml
    python main.py visualize --results results.json --output output.png
    python main.py pretreat --config configs/pretreatment.yaml
"""

import sys
import argparse
import warnings
from pathlib import Path

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
  # Generate training data (use default config)
  python main.py pretreat

  # Use custom config
  python main.py pretreat --config configs/my_pretreatment.yaml

  # Train a model
  python main.py train --config configs/base.yaml

  # Visualize results
  python main.py visualize --results results/output.json --output results/viz.png
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pretreat command (训练数据生成)
    pretreat_parser = subparsers.add_parser("pretreat", help="Generate training data")
    pretreat_parser.add_argument(
        "--config",
        type=str,
        default="configs/pretreatment.yaml",
        help="Path to pretreatment configuration file",
    )
    
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
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Dispatch to appropriate handler
    if args.command == "pretreat":
        run_pretreat(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "visualize":
        run_visualize(args)

def run_pretreat(args):
    """Run training data generation."""
    print("=" * 60)
    print("SAM-Q Training Data Generation")
    print("=" * 60)

    from src.pretreatment.data_generator import TrainingDataGenerator
    from src.utils.config import Config
    from pathlib import Path

    # 加载配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    config = Config(config_path)
    config_dict = config.to_dict()
    print(f"\n配置文件: {config_path}")

    # 从配置字典中提取参数
    data_config = config_dict.get("data", {})
    gen_config = config_dict.get("generation", {})
    aug_config = config_dict.get("augmentation", {})
    
    scene_dir = Path(data_config.get("scene_dir"))
    model_dir = Path(data_config.get("model_dir"))
    output_dir = Path(data_config.get("output_dir", "data"))

    # 验证目录存在
    if not scene_dir.exists():
        print(f"错误: 场景目录不存在: {scene_dir}")
        return
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        return

    print(f"\n配置:")
    print(f"  场景目录: {scene_dir}")
    print(f"  模型目录: {model_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  数据集生成阶段：{gen_config.get('step', 'render_only')}")
    print(f"  图像分辨率: {gen_config.get('image_size', 1024)}")
    print(f"  热力图 Sigma: {gen_config.get('heatmap_sigma', 15.0)}")
    print(f"  数据增强: {'启用' if aug_config.get('enabled', False) else '禁用'}")
    if aug_config.get('enabled', False):
        print(f"  增强比例: {aug_config.get('aug_ratio', 0.5)}")
    print()

    generator = TrainingDataGenerator(config=config_dict)
    generator.run()


def run_train(args):
    """Run training."""
    print("=" * 60)
    print("SAM-Q Training")
    print("=" * 60)

    from src.utils.config import Config
    from src.train import Trainer

    # Load configuration
    config = Config(args.config).to_dict()

    # Override with command line args
    if args.data_dir:
        config["data"]["root_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["save_dir"] = args.output_dir

    data_dir = config.get("data", {}).get("root_dir", "data/")

    # Initialize trainer
    trainer = Trainer(config)

    # Start training (trainer loads data internally)
    trainer.train(data_dir)

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

if __name__ == "__main__":
    main()
