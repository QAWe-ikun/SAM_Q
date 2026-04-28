"""
SAM-Q 训练数据生成入口脚本

从配置文件加载配置，调用 TrainingDataGenerator 执行数据生成。
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pretreatment.data_generator import TrainingDataGenerator

# 配置日志系统
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"generate_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("TrainingDataGenerator")
logger.setLevel(logging.DEBUG)

# 文件处理器
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info(f"加载配置文件: {config_file}")
    return config


def main():
    parser = argparse.ArgumentParser(description="SAM-Q 训练数据生成")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretreatment.yaml",
        help="配置文件路径",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("SAM-Q 训练数据生成")
    logger.info("=" * 60)

    # 加载配置
    config = load_config(args.config)

    # 创建数据生成器
    generator = TrainingDataGenerator(config)

    # 执行数据生成
    generator.run()

    logger.info("数据生成完成!")


if __name__ == "__main__":
    main()
