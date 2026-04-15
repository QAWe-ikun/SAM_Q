"""
Trainer for SAM-Q
==================

Provides comprehensive training loop with validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import yaml
from datetime import datetime

from ..models import SAMQPlacementModel, PlacementLoss
from .optimizer import create_optimizer, create_scheduler
from .metrics import compute_metrics
from ..utils.config import Config


class Trainer:
    """
    Comprehensive trainer for object placement prediction.
    
    Features:
        - Training loop with progress bar
        - Validation loop
        - Checkpoint management
        - Metrics computation
        - Logging (console and optional W&B)
        - Early stopping
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[SAMQPlacementModel] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: Model to train (optional, created if None)
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
        
        # Initialize model
        self.model = model or self._init_model()

        # 根据 loss.type 确定训练阶段
        # stage1 (lm)        → Qwen3-VL LoRA 微调，语言模型损失
        # stage2 (placement) → Adapter + SAM3 Decoder，placement 损失
        # eval / 其他        → 只做评估，不训练
        loss_config = config.get("loss", {})
        self.stage = loss_config.get("type", "placement")  # "lm" | "placement" | "vla"
        print(f"[Trainer] Training stage: {self.stage}")

        # Initialize loss
        if self.stage == "lm":
            # Stage 1: 语言模型损失（cross-entropy on [SEG] token prediction）
            self.criterion = None  # LM loss 在 train_epoch_stage1 里计算
            self._setup_stage1()
        else:
            # Stage 2 / eval: placement loss
            self.criterion = PlacementLoss(
                dice_weight=loss_config.get("dice_weight", 1.0),
                bce_weight=loss_config.get("bce_weight", 1.0),
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
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Early stopping
        self.early_stopping = training_config.get("early_stopping", False)
        self.patience = training_config.get("patience", 20)
        self.patience_counter = 0

    def _init_model(self) -> SAMQPlacementModel:
        """Initialize model from configuration."""
        model_config = self.config.get("model", {})
        num_seg_tokens = model_config.get("num_seg_tokens", 1)

        model = SAMQPlacementModel(
            qwen_model_name=model_config.get("qwen", {}).get(
                "model_name", "Qwen/Qwen3-VL-8B-Instruct"
            ),
            sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
            num_seg_tokens=num_seg_tokens,
            device=self.device,
            seg_token_config=model_config.get("seg_token", {}),
            action_head_config=model_config.get("action_head", {}),
        )
        
        # Freeze components
        if model_config.get("qwen", {}).get("freeze", True):
            model.freeze_qwen()
        if model_config.get("sam3", {}).get("freeze_image_encoder", True):
            model.freeze_sam3_image_encoder()
        
        model.to(self.device)
        return model

    # ------------------------------------------------------------------ #
    # Stage setup helpers
    # ------------------------------------------------------------------ #

    def _setup_stage1(self):
        """Stage 1: 只训练 Qwen3-VL LoRA，冻结其余所有模块。"""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})
        lora_cfg = qwen_cfg.get("lora", {})

        # 确保 SAM3 / Adapter 冻结
        self.model.freeze_sam3_image_encoder()
        if hasattr(self.model, "freeze_sam3_decoder"):
            self.model.freeze_sam3_decoder()
        if hasattr(self.model, "adapter"):
            for p in self.model.adapter.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_projector"):
            for p in self.model.seg_projector.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_action_head"):
            for p in self.model.seg_action_head.parameters():
                p.requires_grad = False

        # 启用 Qwen LoRA
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
        """Stage 2: 冻结 Qwen3-VL，训练 Adapter + SAM3 Decoder。"""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})

        # 冻结 Qwen
        if qwen_cfg.get("freeze", True):
            self.model.freeze_qwen()

        # 加载 Stage 1 LoRA checkpoint（如果有）
        lora_ckpt = qwen_cfg.get("lora_checkpoint", None)
        if lora_ckpt:
            self._load_lora_checkpoint(lora_ckpt)

        # 解冻 SAM3 decoder（如果配置要求）
        sam3_cfg = model_config.get("sam3", {})
        if not sam3_cfg.get("freeze_detector", True):
            self.model._load_sam3()
            for p in self.model.sam3_transformer.parameters():
                p.requires_grad = True
            for p in self.model.sam3_seg_head.parameters():
                p.requires_grad = True
            for p in self.model.sam3_dot_scoring.parameters():
                p.requires_grad = True
            print("[Trainer] Stage 2: SAM3 decoder unfrozen")

        # 解冻 Adapter
        if not model_config.get("adapter", {}).get("freeze", False):
            if hasattr(self.model, "adapter"):
                for p in self.model.adapter.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "seg_projector"):
                for p in self.model.seg_projector.parameters():
                    p.requires_grad = True
            print("[Trainer] Stage 2: Adapter unfrozen")

    def _load_lora_checkpoint(self, lora_ckpt: str):
        """加载 Stage 1 的 LoRA checkpoint。"""
        try:
            from peft import PeftModel
            print(f"[Trainer] Loading LoRA checkpoint from {lora_ckpt}")
            self.model.qwen_encoder.model = PeftModel.from_pretrained(
                self.model.qwen_encoder.model, lora_ckpt
            )
            print("[Trainer] LoRA checkpoint loaded")
        except Exception as e:
            print(f"[Trainer] Warning: failed to load LoRA checkpoint: {e}")

    # ------------------------------------------------------------------ #
    # Stage 1: Language Model training epoch
    # ------------------------------------------------------------------ #

    def train_epoch_stage1(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Stage 1 训练：next-token prediction loss，教 Qwen3-VL 输出 [SEG]。

        Dataset 需要提供 `response` 字段（含 [SEG] 的 GPT 回复）。
        如果 dataset 没有 `response`，自动使用 "好的，我将为您放置物体。[SEG]" 作为目标。
        """
        self.model.qwen_encoder.load_model()
        self.model.train()

        tokenizer = self.model.qwen_encoder.processor.tokenizer
        qwen_model = self.model.qwen_encoder.model

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"[Stage1] Epoch {self.current_epoch + 1}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=self.device)
            plane_images_batch = batch["plane_images"]
            object_images_batch = batch["object_images"]

            for i in range(len(plane_images_batch)):
                plane_img = plane_images_batch[i]
                obj_img = object_images_batch[i]
                text_prompt = batch["text_prompts"][i]
                response = batch.get("responses", [None] * len(plane_images_batch))[i]
                if response is None:
                    response = f"好的，我将为您放置物体。[SEG]"

                # 构造 messages（input + target）
                messages, image_list = self.model.qwen_encoder._build_message(
                    text_prompt=text_prompt,
                    images=[plane_img, obj_img],
                )
                # 拼接 assistant 回复到 messages
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

                text = self.model.qwen_encoder.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                inputs = self.model.qwen_encoder.processor(
                    text=[text],
                    images=image_list if image_list else None,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # 计算 LM loss（labels = input_ids，prefix 部分设为 -100）
                input_ids = inputs["input_ids"]
                labels = input_ids.clone()

                # 找到 assistant 回复起始位置，之前的 token 不计入 loss
                # 用 assistant 标记 token 定位（简化：从最后 N 个 token 开始）
                response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
                resp_len = len(response_tokens)
                labels[:, :-resp_len] = -100  # 只对回复部分计算 loss

                outputs = qwen_model(
                    **inputs,
                    labels=labels,
                )
                loss = outputs.loss
                if loss is not None:
                    batch_loss = batch_loss + loss

            avg_loss = batch_loss / len(plane_images_batch)
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
            self.optimizer.step()

            total_loss += avg_loss.item()
            num_batches += 1

            if batch_idx % log_interval == 0:
                progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}"})

        return {"train_loss": total_loss / max(num_batches, 1)}

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            log_interval: Log every N batches
            
        Returns:
            Metrics dictionary
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            plane_images_batch = batch["plane_images"].to(self.device)
            object_images_batch = batch["object_images"].to(self.device)
            masks = batch["masks"].to(self.device)

            # Process each sample
            batch_loss = 0.0
            for i in range(len(batch["plane_images"])):
                plane_image = plane_images_batch[i]
                object_image = object_images_batch[i]
                text_prompt = batch["text_prompts"][i]

                # Forward pass - new API: images list
                output = self.model(
                    plane_image=plane_image,
                    text_prompt=text_prompt,
                    images=[plane_image, object_image],
                )

                # Compute loss
                loss_dict = self.criterion(
                    output["heatmap"],
                    masks[i:i+1],
                    output.get("rotation_6d"),
                    output.get("scale_relative"),
                    output.get("class_logits"),
                )
                batch_loss += loss_dict["total"]
            
            # Average loss
            avg_loss = batch_loss / len(batch["plane_images"])
            
            # Backward pass
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += avg_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}"})
            
            # Log intermediate metrics
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                    f"Loss: {avg_loss.item():.4f}"
                )
        
        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return {"train_loss": avg_epoch_loss}

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Metrics dictionary
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            plane_images_batch = batch["plane_images"].to(self.device)
            object_images_batch = batch["object_images"].to(self.device)
            masks = batch["masks"].to(self.device)

            batch_loss = 0.0
            for i in range(len(batch["plane_images"])):
                plane_image = plane_images_batch[i]
                object_image = object_images_batch[i]
                text_prompt = batch["text_prompts"][i]

                output = self.model(
                    plane_image=plane_image,
                    text_prompt=text_prompt,
                    images=[plane_image, object_image],
                )

                loss_dict = self.criterion(
                    output["heatmap"],
                    masks[i:i+1],
                    output.get("rotation_6d"),
                    output.get("scale_relative"),
                    output.get("class_logits"),
                )
                batch_loss += loss_dict["total"]

                # Compute metrics
                with torch.no_grad():
                    pred_mask = torch.sigmoid(output["heatmap"])
                    # 选最佳 query（与 PlacementLoss 逻辑一致）
                    class_logits = output.get("class_logits")
                    if pred_mask.shape[1] > 1 and class_logits is not None:
                        scores = class_logits[-1].squeeze(-1)  # [B, num_queries]
                        best_idx = scores.argmax(dim=-1)       # [B]
                        pred_mask = pred_mask[torch.arange(pred_mask.shape[0], device=pred_mask.device), best_idx]
                        pred_mask = pred_mask.unsqueeze(1)     # [B, 1, H, W]
                    # resize 到 GT 尺寸
                    gt_mask = masks[i:i+1]
                    if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                        import torch.nn.functional as F
                        pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)
                    metrics = compute_metrics(pred_mask, gt_mask)
                    for key in all_metrics:
                        all_metrics[key] += metrics[key]
            
            avg_loss = batch_loss / len(batch["plane_images"])
            total_loss += avg_loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        # Average metrics
        for key in all_metrics:
            all_metrics[key] /= num_batches
        
        all_metrics["val_loss"] = avg_val_loss
        
        return all_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
        """
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        val_interval = training_config.get("val_interval", 1)
        log_interval = training_config.get("log_interval", 10)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 根据阶段选择对应的 train_epoch
            if self.stage == "lm":
                train_metrics = self.train_epoch_stage1(train_loader, log_interval)
                val_metrics = {}  # Stage 1 不做 mask 验证
            else:
                # Train
                train_metrics = self.train_epoch(train_loader, log_interval)

                # Validate
                val_metrics = {}
                if val_loader is not None and (epoch + 1) % val_interval == 0:
                    val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            
            # Print metrics
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch"
            )
            print(f"Epoch {metrics['epoch']:3d} | {metrics_str}")
            
            # Save checkpoint
            val_loss = val_metrics.get("val_loss", float("inf"))
            self._save_checkpoint(epoch, val_loss)
            
            # Early stopping
            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
        
        # Save final checkpoint
        self._save_checkpoint("final", float("inf"))
        print(f"\nTraining completed! Results saved to: {self.output_dir}")

    def _save_checkpoint(self, epoch: Any, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }
        
        # Save epoch checkpoint
        if epoch == "final":
            path = self.output_dir / "checkpoint_final.pt"
        else:
            save_interval = self.config.get("training", {}).get("save_interval", 10)
            if isinstance(epoch, int) and (epoch + 1) % save_interval == 0:
                path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best and self.config.get("training", {}).get("save_best", True):
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
