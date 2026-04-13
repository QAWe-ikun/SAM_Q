#!/usr/bin/env python3
"""
Download models for SAM-Q
=========================

Downloads Qwen3-VL-8B and SAM3 to local directory using ModelScope (China mirror).
Supports resuming and skipping already downloaded models.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --force  # Force re-download
"""

import os
import sys
import argparse
from pathlib import Path

def check_modelscope():
    """Check if modelscope is installed."""
    try:
        import modelscope
        print(f"[OK] ModelScope version: {modelscope.__version__}")
        return True
    except ImportError:
        print("[!] ModelScope not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "--upgrade"])
        print("[OK] ModelScope installed.")
        return True


def is_model_downloaded(local_dir, check_file="config.json"):
    """
    Check if model is already downloaded.
    
    Args:
        local_dir: Directory where model should be
        check_file: Key file to verify download completeness
        
    Returns:
        bool: True if model exists and is complete
    """
    path = Path(local_dir)
    
    # Check directory exists
    if not path.exists():
        return False
    
    # Check key file exists
    config_path = path / check_file
    if not config_path.exists():
        # Check subdirectories
        sub_configs = list(path.glob(f"*/**/{check_file}"))
        if not sub_configs:
            return False
    
    return True


def download_model(model_id, local_dir, force=False, source_name="ModelScope"):
    """
    Generic download function using ModelScope.
    """
    print("\n" + "="*60)
    print(f"Checking/Downloading: {model_id}")
    print(f"Source: {source_name}")
    print(f"Target: {local_dir}")
    print("="*60)
    
    # Check if exists
    if not force and is_model_downloaded(local_dir):
        print(f"[OK] Model already exists at {local_dir}")
        print("     Skipping download. (Use --force to re-download)")
        return local_dir
    
    # Download
    try:
        from modelscope import snapshot_download
        
        print("Starting download...")
        # snapshot_download handles caching and resuming automatically
        result_dir = snapshot_download(
            model_id,
            local_dir=local_dir,
            revision='master'
        )
        
        print(f"[OK] Download complete: {result_dir}")
        return result_dir
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("Tip: Check your internet connection or run again to resume.")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download SAM-Q models")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force re-download even if model exists"
    )
    parser.add_argument(
        "--only-qwen",
        action="store_true",
        help="Only download Qwen3-VL"
    )
    parser.add_argument(
        "--only-sam3",
        action="store_true",
        help="Only download SAM3"
    )
    
    args = parser.parse_args()
    
    print("SAM-Q Model Downloader (ModelScope)")
    print("="*60)
    
    # Check dependency
    if not check_modelscope():
        print("[ERROR] Failed to install ModelScope.")
        sys.exit(1)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Download Qwen3-VL
        if not args.only_sam3:
            results['qwen'] = download_model(
                'Qwen/Qwen3-VL-8B-Instruct',
                'models/qwen3_vl',
                force=args.force
            )
        
        # Download SAM3
        # Corrected ID based on user confirmation
        if not args.only_qwen:
            results['sam3'] = download_model(
                'facebook/sam3',  # Correct ModelScope ID
                'models/sam3',
                force=args.force
            )
        
        # Summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        for name, path in results.items():
            print(f"  {name}: {path}")
        
        print("\nTo use local models, update configs/base.yaml:")
        if 'qwen' in results:
            print(f'  model.qwen.model_name: "{results["qwen"]}"')
            
    except Exception as e:
        print(f"\n[ERROR] Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
