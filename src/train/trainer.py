"""
Trainer for SAM-Q
==================

Main entry point that coordinates Stage 1 and Stage 2 training.
"""

import yaml
import torch # type: ignore
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from .components import (
    create_dataloaders,
    split_dataloaders,
    Stage1Trainer,
    Stage2Trainer,
    SegFeatureExtractor,
)
from .optimizer import create_optimizer, create_scheduler


class Trainer:
    """
    Main trainer that coordinates the full training pipeline.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            device: Device to run on
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get("experiment", {}).get("name", "experiment")
        self.output_dir = Path(config.get("training", {}).get("save_dir", "outputs"))
        self.output_dir = self.output_dir / f"{exp_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Determine training stage
        loss_config = config.get("loss", {})
        self.stage = loss_config.get("type", "placement")  # "lm" | "placement"

        # Initialize model
        self.model = self._init_model()

        # Initialize loss
        if self.stage == "lm":
            self.criterion = None
            self._setup_stage1()
        else:
            from ..models import PlacementLoss
            self.criterion = PlacementLoss(
                dice_weight=loss_config.get("dice_weight", 1.0),
                bce_weight=loss_config.get("bce_weight", 1.0),
                rotation_weight=loss_config.get("rotation_weight", 0.5),
                scale_weight=loss_config.get("scale_weight", 0.3),
            ).to(self.device)
            self._setup_stage2()

        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            config.get("optimizer", {})
        )

        training_config = config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        scheduler_config = config.get("scheduler", {})
        warmup_epochs = scheduler_config.get("warmup_epochs", 0)
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_config,
            num_epochs,
            warmup_epochs,
        )

    def _init_model(self):
        """Initialize model from configuration."""
        from ..models import SAMQPlacementModel

        model_config = self.config.get("model", {})
        num_seg_tokens = model_config.get("num_seg_tokens", 1)

        data_config = self.config.get("data", {})
        seg_feature_dir = data_config.get("seg_feature_dir")
        use_qwen = self.stage != "placement" or seg_feature_dir is None

        if self.stage == "placement" and seg_feature_dir is not None:
            print(f"[Trainer] Stage 2 with seg_features: skipping Qwen3-VL (saves ~16GB VRAM)")
        elif self.stage == "placement":
            print(f"[Trainer] Stage 2 without seg_features: loading Qwen3-VL for online inference")

        model = SAMQPlacementModel(
            sam_checkpoint_path=model_config.get("sam3", {}).get("sam_checkpoint_path", None),
            adapter_checkpoint_path=model_config.get("adapter", {}).get("adapter_checkpoint_path", None),
            qwen_model_name=model_config.get("qwen", {}).get("model_name") if use_qwen else None,
            qwen_lora_path=model_config.get("qwen", {}).get("lora_path", None) if use_qwen else None,
            sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
            num_seg_tokens=num_seg_tokens,
            device=self.device,
            seg_token_config=model_config.get("seg_token", {}),
            action_head_config=model_config.get("action_head", {}),
        )

        # Freeze components
        if model.qwen_encoder is not None and model_config.get("qwen", {}).get("freeze", True):
            model.freeze_qwen()
        if model.sam3_loader is not None and model_config.get("sam3", {}).get("freeze_image_encoder", True):
            model.freeze_sam3_image_encoder()

        model.to(self.device)
        return model

    def _setup_stage1(self):
        """Stage 1: Enable Qwen3-VL LoRA, freeze rest."""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})
        lora_cfg = qwen_cfg.get("lora", {})

        # Freeze SAM3/Adapter if they exist
        if self.model.sam3_loader is not None:
            self.model.freeze_sam3_image_encoder()
        if hasattr(self.model, "adapter") and self.model.adapter is not None:
            for p in self.model.adapter.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_projector") and self.model.seg_projector is not None:
            for p in self.model.seg_projector.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_action_head") and self.model.seg_action_head is not None:
            for p in self.model.seg_action_head.parameters():
                p.requires_grad = False

        # Enable LoRA
        if lora_cfg.get("enabled", True):
            self.model.qwen_encoder.enable_finetuning(
                lora_r=lora_cfg.get("r", 64),
                lora_alpha=lora_cfg.get("alpha", 128),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                target_modules=lora_cfg.get("target_modules", None),
                use_qlora=lora_cfg.get("use_qlora", False),
                lora_bias=lora_cfg.get("bias", "none"),
            )
            print("[Trainer] Stage 1: Qwen3-VL LoRA enabled")

    def _setup_stage2(self):
        """Stage 2: Freeze Qwen, unfreeze Adapter + SAM3 Decoder."""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})

        # Freeze Qwen
        if qwen_cfg.get("freeze", True):
            self.model.freeze_qwen()

        # Load Stage 1 LoRA checkpoint
        lora_ckpt = qwen_cfg.get("lora_checkpoint", None)
        if lora_ckpt:
            self._load_lora_checkpoint(lora_ckpt)

        # Unfreeze SAM3 decoder parts
        sam3_cfg = model_config.get("sam3", {})
        if not sam3_cfg.get("freeze_detector", True):
            self.model.sam3_loader.load_model()

            for p in self.model.sam3_loader.sam3_transformer.decoder.parameters():
                p.requires_grad = True
            for p in self.model.sam3_loader.sam3_seg_head.parameters():
                p.requires_grad = True
            for p in self.model.sam3_loader.sam3_dot_scoring.parameters():
                p.requires_grad = True

            # Keep frozen
            for p in self.model.sam3_loader.model.backbone.language_backbone.parameters():
                p.requires_grad = False
            for p in self.model.sam3_loader.model.geometry_encoder.parameters():
                p.requires_grad = False
            for p in self.model.sam3_loader.model.transformer.encoder.parameters():
                p.requires_grad = False

            print("[Trainer] Stage 2: SAM3 decoder unfrozen (decoder + seg_head + dot_scoring)")

        # Unfreeze Adapter
        if not model_config.get("adapter", {}).get("freeze", False):
            if hasattr(self.model, "adapter") and self.model.adapter is not None:
                for p in self.model.adapter.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "seg_projector") and self.model.seg_projector is not None:
                for p in self.model.seg_projector.parameters():
                    p.requires_grad = True
            print("[Trainer] Stage 2: Adapter unfrozen")

        # Check seg features
        seg_dir = self.config.get("data", {}).get("seg_feature_dir")
        if seg_dir:
            print(f"[Trainer] 使用预提取 <SEG> features: {seg_dir}")

        print(f"\n{'='*60}")
        print(f"[Stage 2] Trainable Parameters Check:")
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Stage 2] Total Trainable Params: {total_trainable:,}")
        print(f"{'='*60}\n")

    def _load_lora_checkpoint(self, lora_ckpt: str):
        """Load Stage 1 LoRA checkpoint."""
        try:
            from peft import PeftModel  # type: ignore
            print(f"[Trainer] Loading LoRA checkpoint from {lora_ckpt}")
            self.model.qwen_encoder.model = PeftModel.from_pretrained(
                self.model.qwen_encoder.model, lora_ckpt
            )
            print("[Trainer] LoRA checkpoint loaded")
        except Exception as e:
            print(f"[Trainer] Warning: failed to load LoRA checkpoint: {e}")

    def train(self, data_dir: str) -> None:
        """
        Full training loop. Loads data internally.

        Args:
            data_dir: Root directory containing train/val/test splits
        """
        # Load DataLoaders
        data_config = self.config.get("data", {})
        seg_feature_dir = data_config.get("seg_feature_dir")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir, self.config, seg_feature_dir
        )

        # Print dataset info
        print(f"\nDataset info:")
        print(f"  Training samples: {len(train_loader.dataset) if train_loader else 0}")
        print(f"  Validation samples: {len(val_loader.dataset) if val_loader else 0}")
        print(f"  Test samples: {len(test_loader.dataset) if test_loader else 0}")
        print(f"  Data directory: {data_dir}")

        # Apply sample limit for testing
        max_samples = data_config.get("max_samples", None)
        train_loader, val_loader, test_loader = split_dataloaders(
            train_loader, val_loader, test_loader, max_samples
        )

        # Route to appropriate stage
        if self.stage == "lm":
            self._run_stage1(train_loader, val_loader, test_loader)
        else:
            self._run_stage2(train_loader, val_loader, test_loader)

    def _run_stage1(self, train_loader, val_loader, test_loader):
        """Run Stage 1 training."""
        stage1 = Stage1Trainer(self.model, self.config, self.output_dir, self.device)

        # Train
        stage1.train(train_loader)

        # Validate
        if val_loader is not None:
            print(f"\n{'='*60}")
            print(f"Running Stage 1 Validation...")
            stage1.validate(val_loader)
            print(f"{'='*60}\n")

        # Extract seg features
        seg_dir = Path(self.config.get("data", {}).get("root_dir", "data/")) / "seg_features"
        seg_extractor = SegFeatureExtractor(self.model, self.config)
        seg_extractor.extract(train_loader, seg_dir)
        if val_loader is not None:
            seg_extractor.extract(val_loader, seg_dir)
        if test_loader is not None:
            seg_extractor.extract(test_loader, seg_dir)

        # Test evaluation
        data_config = self.config.get("data", {})
        max_samples = data_config.get("max_samples", None)
        if test_loader is not None and max_samples is None:
            print(f"\n{'='*60}")
            print(f"Running Stage 1 Test Evaluation...")
            stage1.validate(test_loader)
            print(f"{'='*60}\n")

        print(f"\nStage 1 training completed!")

    def _run_stage2(self, train_loader, val_loader, test_loader):
        """Run Stage 2 training."""
        stage2 = Stage2Trainer(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            output_dir=self.output_dir,
            device=self.device,
        )

        # Train
        stage2.train(train_loader, val_loader)

        # Test evaluation
        data_config = self.config.get("data", {})
        max_samples = data_config.get("max_samples", None)
        if test_loader is not None and max_samples is None:
            print(f"\n{'='*60}")
            print(f"Running Stage 2 Test Evaluation...")
            stage2._validate(test_loader)
            print(f"{'='*60}\n")

        print(f"\nStage 2 training completed!")
