"""
Data Splitter
=============

Splits DataLoaders for quick testing with limited samples.
"""

from typing import Optional, Tuple
from torch.utils.data import DataLoader, Subset  # type: ignore


def split_dataloaders(
    train_loader: Optional[DataLoader],
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    max_samples: Optional[int] = None,
    train_ratio: float = 0.8,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Split DataLoaders to limit total samples for quick testing.

    Args:
        train_loader: Original train DataLoader
        val_loader: Original val DataLoader
        test_loader: Original test DataLoader
        max_samples: Maximum total samples to use (None = use all)
        train_ratio: Ratio of samples for training (rest goes to val)

    Returns:
        Tuple of (train_loader, val_loader, test_loader) with limited samples
    """
    if max_samples is None or train_loader is None:
        return train_loader, val_loader, test_loader

    full_dataset = train_loader.dataset
    total = len(full_dataset)

    if max_samples >= total:
        return train_loader, val_loader, test_loader

    # Calculate split sizes
    n_train = int(max_samples * train_ratio)
    n_val = max_samples - n_train

    # Create subsets
    train_ds = Subset(full_dataset, range(n_train))
    val_ds = Subset(full_dataset, range(n_train, n_train + n_val))

    # Rebuild DataLoaders
    new_train_loader = DataLoader(
        train_ds,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        collate_fn=train_loader.collate_fn,
    )

    new_val_loader = None
    if val_loader is not None:
        new_val_loader = DataLoader(
            val_ds,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            collate_fn=val_loader.collate_fn,
        )

    # Test loader remains unchanged (not part of the split)
    print(f"[DataSplitter] 使用 {max_samples} 条样本进行训练: train={n_train}, val={n_val}")

    return new_train_loader, new_val_loader, test_loader
