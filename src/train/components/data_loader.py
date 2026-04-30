"""
Data Loader Factory
====================

Creates Dataset and DataLoader instances for training, validation, and test splits.
"""

from typing import Dict, Any, Optional, Tuple
from torch.utils.data import DataLoader  # type: ignore


def create_dataloaders(
    data_dir: str,
    config: Dict[str, Any],
    seg_feature_dir: Optional[str] = None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train/val/test DataLoaders.

    Args:
        data_dir: Root directory containing train/val/test splits
        config: Full training configuration
        seg_feature_dir: Optional path to pre-extracted seg features

    Returns:
        Tuple of (train_loader, val_loader, test_loader), any can be None
    """
    from src.data.dataset import ObjectPlacementDataset

    batch_size = config.get("training", {}).get("batch_size", 2)
    num_workers = config.get("data", {}).get("num_workers", 4)

    loaders = []

    for split in ["train", "val", "test"]:
        try:
            dataset = ObjectPlacementDataset(
                data_dir=data_dir,
                split=split,
                seg_feature_dir=seg_feature_dir,
            )

            if len(dataset) == 0:
                print(f"  [Warning] {split} dataset is empty, skipping")
                loaders.append(None)
                continue

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=dataset._collate_fn if hasattr(dataset, '_collate_fn') else None,
            )
            loaders.append(loader)

        except Exception as e:
            print(f"  [Warning] Failed to load {split} dataset: {e}")
            loaders.append(None)

    return loaders[0], loaders[1], loaders[2]
