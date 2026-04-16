"""
Trainer for SAM-Q
==================

Provides comprehensive training loop with validation, checkpointing, and logging.
"""

import yaml
import torch
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List

from .metrics import compute_metrics
from ..models import SAMQPlacementModel, PlacementLoss
from .optimizer import create_optimizer, create_scheduler


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

        # 根据 loss.type 确定训练阶段
        # stage1 (lm)        → Qwen3-VL LoRA 微调, 语言模型损失
        # stage2 (placement) → Adapter + SAM3 Decoder, placement 损失
        loss_config = config.get("loss", {})
        self.stage = loss_config.get("type", "placement")  # "lm" | "placement"
        print(f"[Trainer] Training stage: {self.stage}")

        # Initialize model (needs self.stage)
        self.model = model or self._init_model()

        # Initialize loss
        if self.stage == "lm":
            # Stage 1: 语言模型损失（cross-entropy on <SEG> token prediction）
            self.criterion = None  # LM loss 在 train_epoch_stage1 里计算
            self._setup_stage1()
        else:
            # Stage 2: placement loss
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

        # Stage 1 (LM) does not need SAM3; Stage 2 (placement) requires it
        sam3_ckpt = model_config.get("sam3", {}).get("checkpoint_path")
        if self.stage == "lm":
            sam3_ckpt = None
        print(f"[Trainer] Initializing model with SAM3 checkpoint: {sam3_ckpt}")

        # Stage 2 with pre-extracted seg_features doesn't need Qwen3-VL
        # This saves ~16GB VRAM
        data_config = self.config.get("data", {})
        seg_feature_dir = data_config.get("seg_feature_dir")
        use_qwen = self.stage != "placement" or seg_feature_dir is None

        if self.stage == "placement" and seg_feature_dir is not None:
            print(f"[Trainer] Stage 2 with seg_features: skipping Qwen3-VL (saves ~16GB VRAM)")
        elif self.stage == "placement":
            print(f"[Trainer] Stage 2 without seg_features: loading Qwen3-VL for online inference")

        model = SAMQPlacementModel(
            sam3_checkpoint_path=sam3_ckpt,
            qwen_model_name=model_config.get("qwen", {}).get("model_name") if use_qwen else None,
            qwen_lora_path=model_config.get("qwen", {}).get("lora_path") if use_qwen else None,
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
        # Only freeze SAM3 if it was loaded (i.e. not None)
        if model.sam3_loader is not None and model_config.get("sam3", {}).get("freeze_image_encoder", True):
            model.freeze_sam3_image_encoder()
        
        model.to(self.device)
        return model

    # ------------------------------------------------------------------ #
    # Stage setup helpers
    # ------------------------------------------------------------------ #

    def _setup_stage1(self):
        """Stage 1: 只训练 Qwen3-VL LoRA, 冻结其余所有模块。"""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})
        lora_cfg = qwen_cfg.get("lora", {})

        # 确保 SAM3 / Adapter 冻结 (if they exist)
        if self.model.sam3_loader is not None:
            self.model.freeze_sam3_image_encoder()
            if hasattr(self.model, "freeze_sam3_decoder"):
                self.model.freeze_sam3_decoder()
        if hasattr(self.model, "adapter") and self.model.adapter is not None:
            for p in self.model.adapter.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_projector") and self.model.seg_projector is not None:
            for p in self.model.seg_projector.parameters():
                p.requires_grad = False
        if hasattr(self.model, "seg_action_head") and self.model.seg_action_head is not None:
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
        """Stage 2: 冻结 Qwen3-VL, 训练 Adapter + SAM3 Decoder。"""
        model_config = self.config.get("model", {})
        qwen_cfg = model_config.get("qwen", {})

        # 冻结 Qwen
        if qwen_cfg.get("freeze", True):
            self.model.freeze_qwen()

        # 加载 Stage 1 LoRA checkpoint（如果有）
        lora_ckpt = qwen_cfg.get("lora_checkpoint", None)
        if lora_ckpt:
            self._load_lora_checkpoint(lora_ckpt)

        # 解冻 SAM3 decoder（精确控制）
        sam3_cfg = model_config.get("sam3", {})
        if not sam3_cfg.get("freeze_detector", True):
            self.model.sam3_loader.load_model()
            
            # ✅ 训练 transformer decoder（用于 heatmap 生成）
            for p in self.model.sam3_loader.sam3_transformer.decoder.parameters():
                p.requires_grad = True
                
            # ✅ 训练 segmentation head
            for p in self.model.sam3_loader.sam3_seg_head.parameters():
                p.requires_grad = True
                
            # ✅ 训练 dot product scoring
            for p in self.model.sam3_loader.sam3_dot_scoring.parameters():
                p.requires_grad = True
                
            # ❌ 保持冻结：language backbone, geometry encoder, transformer encoder
            for p in self.model.sam3_loader.model.backbone.language_backbone.parameters():
                p.requires_grad = False
            for p in self.model.sam3_loader.model.geometry_encoder.parameters():
                p.requires_grad = False
            for p in self.model.sam3_loader.model.transformer.encoder.parameters():
                p.requires_grad = False
                
            print("[Trainer] Stage 2: SAM3 decoder unfrozen (only decoder + seg_head + dot_scoring)")

        # 解冻 Adapter
        if not model_config.get("adapter", {}).get("freeze", False):
            if hasattr(self.model, "adapter") and self.model.adapter is not None:
                for p in self.model.adapter.parameters():
                    p.requires_grad = True
            if hasattr(self.model, "seg_projector") and self.model.seg_projector is not None:
                for p in self.model.seg_projector.parameters():
                    p.requires_grad = True
            print("[Trainer] Stage 2: Adapter unfrozen")

        # 检测是否使用预提取 seg features
        seg_dir = self.config.get("data", {}).get("seg_feature_dir")
        if seg_dir:
            print(f"[Trainer] 使用预提取 <SEG> features: {seg_dir}, 跳过 Qwen3-VL")
            
        print(f"\n{'='*60}")
        print(f"[Stage 2] Trainable Parameters Check:")
        total_trainable = 0
        for _, param in self.model.named_parameters():
            if param.requires_grad:
                total_trainable += param.numel()
        print(f"[Stage 2] Total Trainable Params: {total_trainable:,}")
        print(f"{'='*60}\n")

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

    def run_sft_stage1(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Stage 1 训练入口: 使用 HuggingFace SFTTrainer 微调 Qwen3-VL。
        
        该方法会接管整个训练流程（包含所有 Epochs），不需要外部循环。
        """
        try:
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except ImportError:
            raise ImportError(
                "Please install trl: pip install trl>=0.8.0"
            )

        training_config = self.config.get("training", {})
        qwen_cfg = self.config.get("model", {}).get("qwen", {})
        lora_cfg = qwen_cfg.get("lora", {})

        # Ensure model is loaded with LoRA
        self.model.qwen_encoder.load_model(use_cache=False)
        if not self.model.qwen_encoder.training_mode:
            self.model.qwen_encoder.enable_finetuning(
                lora_r=lora_cfg.get("r", 64),
                lora_alpha=lora_cfg.get("alpha", 128),
                lora_dropout=lora_cfg.get("dropout", 0.05),
                use_qlora=lora_cfg.get("use_qlora", False),
            )

        qwen_model = self.model.qwen_encoder.model
        tokenizer = self.model.qwen_encoder.processor.tokenizer

        # Configure training arguments
        grad_accum = training_config.get("gradient_accumulation_steps", 1)
        batch_size = training_config.get("batch_size", 1)
        num_epochs = training_config.get("num_epochs", 3)
        lr = self.config.get("optimizer", {}).get("lr", 2e-4)
        warmup_steps = self.config.get("scheduler", {}).get("warmup_epochs", 0)
        save_steps = training_config.get("save_interval", 500)
        log_steps = training_config.get("log_interval", 10)

        # Determine dtype
        use_bf16 = self.config.get("training", {}).get("bf16", False)
        use_fp16 = self.config.get("training", {}).get("fp16", True)

        # Use TrainingArguments instead of SFTConfig to avoid version conflicts
        # SFTTrainer works fine with TrainingArguments when using a custom collator
        sft_config = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=log_steps,
            save_strategy="no",  # Disable SFTTrainer's automatic checkpoint saving
            report_to="none",
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=self.config.get("optimizer", {}).get("weight_decay", 0.01),
            max_grad_norm=1.0,
            remove_unused_columns=False,
        )

        # Create data collator for Qwen3-VL format
        def qwen_data_collator(examples):
            """Custom collator that converts batch to Qwen3-VL input format."""
            texts = []
            images = []

            for ex in examples:
                text_prompt = ex.get("text_prompt", "")
                response = ex.get("response", "好的，我将为您放置物体。<SEG>")

                # Build messages
                messages, img_list = self.model.qwen_encoder._build_message(
                    text_prompt=text_prompt,
                    images=ex.get("images", []),
                )
                messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

                # Apply chat template
                text = self.model.qwen_encoder.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
                images.append(img_list if img_list else None)

            # Process with Qwen3-VL processor
            inputs = self.model.qwen_encoder.processor(
                text=texts,
                images=images if any(img is not None for img in images) else None,
                return_tensors="pt",
                padding=True,
            )

            # Create labels for LM loss
            labels = inputs["input_ids"].clone()

            # Mask out prompt tokens: only compute loss on the Assistant response
            for i, ex in enumerate(examples):
                response = ex.get("response", "")
                # 计算 response 的 token 长度
                response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
                resp_len = len(response_tokens)

                # 获取该样本的实际长度（不含 padding）
                real_len = inputs["attention_mask"][i].sum().item()

                # 将 response 之前的所有 token 设为 -100 (ignore index)
                # 注意：response 总是位于序列末尾（在生成文本之前）
                if real_len > resp_len:
                    labels[i, :real_len - resp_len] = -100
                else:
                    # 极端情况：response 比实际长度还长（通常是因为 special tokens 差异）
                    # 这种情况下至少要把非 response 的部分 mask 掉
                    # 为了安全，我们假设回复在最后，前面的都 mask
                    pass 

                # 确保 padding 部分也是 -100
                labels[i, real_len:] = -100
            
            inputs["labels"] = labels
            return inputs

        # Initialize SFTTrainer
        # Note: We use a custom data_collator, so passing 'tokenizer' is not strictly required
        # and avoids compatibility issues with different trl versions.
        trainer = SFTTrainer(
            model=qwen_model,
            train_dataset=dataloader.dataset,
            data_collator=qwen_data_collator,
            args=sft_config,
        )

        # Train
        print(f"\n{'='*60}")
        print(f"Starting Stage 1 training with SFTTrainer")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {batch_size * grad_accum}")
        print(f"  Learning rate: {lr}")
        print(f"{'='*60}\n")

        train_result = trainer.train()

        # Save LoRA weights
        lora_output_dir = self.output_dir / "lora_weights"
        qwen_model.save_pretrained(lora_output_dir)
        print(f"\nLoRA weights saved to {lora_output_dir}")

        # Sync model state back to wrapper
        self.model.qwen_encoder.model = qwen_model

        return {"train_loss": train_result.metrics.get("train_loss", 0.0)}

    def train_epoch_stage2(
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
        total_bce_loss = 0.0
        total_dice_loss = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            plane_images_batch = batch["plane_images"].to(self.device)
            batch_images = batch["images"]  # List[List[Tensor]]
            masks = batch["masks"].to(self.device)
            seg_hidden_batch = batch.get("seg_hidden")  # [B, hidden_dim] or None
            if seg_hidden_batch is not None:
                seg_hidden_batch = seg_hidden_batch.to(self.device)

            # Process each sample
            batch_loss_tensor = torch.tensor(0.0, device=self.device)
            batch_loss = 0.0
            batch_bce = 0.0
            batch_dice = 0.0
            batch_rot = 0.0
            batch_scl = 0.0

            for i in range(len(plane_images_batch)):
                plane_image = plane_images_batch[i]
                sample_images = [img.to(self.device) for img in batch_images[i]]
                text_prompt = batch["text_prompts"][i]
                seg_hidden_i = seg_hidden_batch[i] if seg_hidden_batch is not None else None

                # Forward pass
                output = self.model(
                    plane_image=plane_image,
                    text_prompt=text_prompt,
                    images=sample_images if seg_hidden_i is None else None,
                    seg_hidden=seg_hidden_i,
                )

                # Compute loss
                gt_rot = batch.get("rotation_6d")
                gt_scl = batch.get("scale")
                loss_dict = self.criterion(
                    predicted_masks=output["heatmap"],
                    target_masks=masks[i:i+1],
                    pred_rotation_6d=output.get("rotation_6d"),
                    pred_scale=output.get("scale_relative"),
                    gt_rotation_6d=gt_rot[i:i+1] if gt_rot is not None else None,
                    gt_scale=gt_scl[i:i+1] if gt_scl is not None else None,
                    class_logits=output.get("class_logits"),
                )
                batch_loss_tensor = batch_loss_tensor + loss_dict["total"]
                batch_loss += loss_dict["total"].item()
                batch_bce += loss_dict.get("bce", 0).item() if isinstance(loss_dict.get("bce", 0), torch.Tensor) else loss_dict.get("bce", 0)
                batch_dice += loss_dict.get("dice", 0).item() if isinstance(loss_dict.get("dice", 0), torch.Tensor) else loss_dict.get("dice", 0)
                batch_rot += loss_dict.get("rotation", 0).item() if isinstance(loss_dict.get("rotation", 0), torch.Tensor) else loss_dict.get("rotation", 0)
                batch_scl += loss_dict.get("scale", 0).item() if isinstance(loss_dict.get("scale", 0), torch.Tensor) else loss_dict.get("scale", 0)

            # Average loss
            avg_loss = batch_loss_tensor / len(batch["plane_images"])
            avg_bce = batch_bce / len(batch["plane_images"])
            avg_dice = batch_dice / len(batch["plane_images"])
            avg_rot = batch_rot / len(batch["plane_images"])
            avg_scl = batch_scl / len(batch["plane_images"])

            # Backward pass
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += avg_loss
            total_bce_loss += avg_bce
            total_dice_loss += avg_dice
            total_rotation_loss += avg_rot
            total_scale_loss += avg_scl
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log intermediate metrics
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                    f"Loss: {avg_loss:.4f}"
                )

        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)

        return {
            "train_loss": avg_epoch_loss,
            "train_bce_loss": total_bce_loss / num_batches,
            "train_dice_loss": total_dice_loss / num_batches,
            "train_rotation_loss": total_rotation_loss / num_batches,
            "train_scale_loss": total_scale_loss / num_batches,
        }
    
    def _validate_stage1(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Stage 1 验证：计算 LM Loss 并生成回复文本。
        """
        self.model.qwen_encoder.load_model()
        self.model.eval()

        tokenizer = self.model.qwen_encoder.processor.tokenizer
        qwen_model = self.model.qwen_encoder.model

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="[Val] Stage1", leave=False)):
                batch_loss = 0.0
                batch_images = batch["images"]
                batch_size = len(batch_images)

                for i in range(batch_size):
                    sample_images = [img.to(self.device) for img in batch_images[i]]
                    text_prompt = batch["text_prompts"][i]
                    response = batch.get("responses", [None] * batch_size)[i]
                    if response is None:
                        response = "好的，我将为您放置物体。<SEG>"

                    # 构造输入
                    messages, image_list = self.model.qwen_encoder._build_message(
                        text_prompt=text_prompt,
                        images=sample_images,
                    )
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

                    # 计算 Loss
                    input_ids = inputs["input_ids"]
                    labels = input_ids.clone()
                    response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
                    resp_len = len(response_tokens)
                    labels[:, :-resp_len] = -100

                    outputs = qwen_model(
                        **inputs,
                        labels=labels,
                    )
                    loss = outputs.loss
                    if loss is not None:
                        batch_loss += loss.item()

                total_loss += batch_loss / batch_size
                num_batches += 1

                # 在第一个 batch 生成样本并打印
                if batch_idx == 0:
                    print(f"\n{'='*60}")
                    print(f"[Stage1 Validation Samples] (Epoch {self.current_epoch+1})")
                    print(f"{'='*60}")

                    # 生成样本（前 2 个）
                    num_gen_samples = min(2, batch_size)
                    for i in range(num_gen_samples):
                        sample_images = [img.to(self.device) for img in batch["images"][i]]
                        text_prompt = batch["text_prompts"][i]
                        response_gt = batch.get("responses", [None] * batch_size)[i]

                        # 构造仅含 prompt 的输入
                        messages, image_list = self.model.qwen_encoder._build_message(
                            text_prompt=text_prompt,
                            images=sample_images,
                        )
                        
                        text = self.model.qwen_encoder.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True,
                        )
                        inputs = self.model.qwen_encoder.processor(
                            text=[text],
                            images=image_list if image_list else None,
                            return_tensors="pt",
                            padding=True,
                        ).to(self.device)

                        # 生成回复
                        generated_ids = qwen_model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                        )
                        
                        # 解码并打印
                        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
                        # 提取 assistant 部分
                        assistant_marker = "assistant"
                        if assistant_marker in generated_text:
                            generated_text = generated_text.split(assistant_marker)[-1]

                        print(f"Prompt:   {text_prompt[:60]}...")
                        print(f"Generated: {generated_text.strip()}")
                        print(f"Expected:  {response_gt.strip() if response_gt else 'N/A'}")
                        print(f"{'-'*60}")

        return {"val_loss": total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def _validate_stage2(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Stage 2 验证：计算 IoU, Precision, Recall, F1 等指标。
        """
        self.model.eval()

        total_loss = 0.0
        total_bce_loss = 0.0
        total_dice_loss = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0
        all_metrics = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        for batch in tqdm(dataloader, desc="Validation", leave=False):
            plane_images_batch = batch["plane_images"].to(self.device)
            batch_images = batch["images"]  # List[List[Tensor]]
            masks = batch["masks"].to(self.device)
            seg_hidden_batch = batch.get("seg_hidden")
            if seg_hidden_batch is not None:
                seg_hidden_batch = seg_hidden_batch.to(self.device)

            batch_loss = 0.0
            batch_bce = 0.0
            batch_dice = 0.0
            batch_rot = 0.0
            batch_scl = 0.0
            
            for i in range(len(plane_images_batch)):
                plane_image = plane_images_batch[i]
                sample_images = [img.to(self.device) for img in batch_images[i]]
                text_prompt = batch["text_prompts"][i]
                seg_hidden_i = seg_hidden_batch[i] if seg_hidden_batch is not None else None

                output = self.model(
                    plane_image=plane_image,
                    text_prompt=text_prompt,
                    images=sample_images if seg_hidden_i is None else None,
                    seg_hidden=seg_hidden_i,
                )

                gt_rot = batch.get("rotation_6d")
                gt_scl = batch.get("scale")
                loss_dict = self.criterion(
                    predicted_masks=output["heatmap"],
                    target_masks=masks[i:i+1],
                    pred_rotation_6d=output.get("rotation_6d"),
                    pred_scale=output.get("scale_relative"),
                    gt_rotation_6d=gt_rot[i:i+1] if gt_rot is not None else None,
                    gt_scale=gt_scl[i:i+1] if gt_scl is not None else None,
                    class_logits=output.get("class_logits"),
                )
                batch_loss += loss_dict["total"].item()
                batch_bce += loss_dict.get("bce", 0).item() if isinstance(loss_dict.get("bce", 0), torch.Tensor) else loss_dict.get("bce", 0)
                batch_dice += loss_dict.get("dice", 0).item() if isinstance(loss_dict.get("dice", 0), torch.Tensor) else loss_dict.get("dice", 0)
                batch_rot += loss_dict.get("rotation", 0).item() if isinstance(loss_dict.get("rotation", 0), torch.Tensor) else loss_dict.get("rotation", 0)
                batch_scl += loss_dict.get("scale", 0).item() if isinstance(loss_dict.get("scale", 0), torch.Tensor) else loss_dict.get("scale", 0)

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
            avg_bce = batch_bce / len(batch["plane_images"])
            avg_dice = batch_dice / len(batch["plane_images"])
            avg_rot = batch_rot / len(batch["plane_images"])
            avg_scl = batch_scl / len(batch["plane_images"])
            
            total_loss += avg_loss
            total_bce_loss += avg_bce
            total_dice_loss += avg_dice
            total_rotation_loss += avg_rot
            total_scale_loss += avg_scl
            num_batches += 1

        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)

        # Average metrics
        for key in all_metrics:
            all_metrics[key] /= num_batches

        all_metrics["val_loss"] = avg_val_loss
        all_metrics["val_bce_loss"] = total_bce_loss / num_batches
        all_metrics["val_dice_loss"] = total_dice_loss / num_batches
        all_metrics["val_rotation_loss"] = total_rotation_loss / num_batches
        all_metrics["val_scale_loss"] = total_scale_loss / num_batches
        
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
        # Stage 1: SFTTrainer handles its own loop
        if self.stage == "lm":
            self.run_sft_stage1(train_loader)

            # Run validation to check generation quality
            if val_loader is not None:
                print(f"\n{'='*60}")
                print(f"Running Stage 1 Validation (Check <SEG> generation)...")
                self._validate_stage1(val_loader)
                print(f"{'='*60}\n")

            # SFTTrainer 训练结束后，尝试提取 <SEG> 特征
            seg_dir = Path(self.config.get("data", {}).get("root_dir", "data/")) / "seg_features"
            self._extract_seg_features(train_loader, seg_dir)
            if val_loader is not None:
                self._extract_seg_features(val_loader, seg_dir)
            print(f"\nStage 1 training completed!")
            return

        # Stage 2: Custom Epoch loop
        self.train_stage2(train_loader, val_loader)
        print(f"\n{'='*60}")
        print(f"Stage 2 training completed!")
        print(f"{'='*60}\n")

    def train_stage2(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Stage 2 Training: Adapter + SAM3 Decoder.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
        """
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        val_interval = training_config.get("val_interval", 1)
        log_interval = training_config.get("log_interval", 10)

        print(f"\n{'='*60}")
        print(f"Starting Stage 2 Training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch_stage2(train_loader, log_interval)

            # Validate
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_metrics = self._validate_stage2(val_loader)

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
            epoch_num = metrics['epoch']
            print(f"Epoch {epoch_num:3d} | "
                  f"train_loss: {metrics.get('train_loss', 0):.4f} | "
                  f"bce: {metrics.get('train_bce_loss', 0):.4f} | "
                  f"dice: {metrics.get('train_dice_loss', 0):.4f} | "
                  f"rot: {metrics.get('train_rotation_loss', 0):.4f} | "
                  f"scl: {metrics.get('train_scale_loss', 0):.4f} | "
                  f"val_loss: {metrics.get('val_loss', 0):.4f} | "
                  f"iou: {metrics.get('iou', 0):.4f} | "
                  f"lr: {metrics.get('lr', 0):.6f}")

            # Determine if this is the best model so far
            val_loss = val_metrics.get("val_loss", float("inf"))
            is_best = val_loss < self.best_val_loss

            # Update best_val_loss immediately
            if is_best:
                self.best_val_loss = val_loss

            # Early stopping
            if self.early_stopping:
                if is_best:
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break

            # Save checkpoint
            self._save_checkpoint(epoch, val_loss, is_best)

        # Save final checkpoint
        self._save_checkpoint("final", float("inf"))
        
        # If Stage 2, ensure split weights are also saved for the final checkpoint
        training_config = self.config.get("training", {})
        if self.stage == "placement" and training_config.get("save_split_weights", True):
            self._save_split_checkpoint(suffix="_final")
            
        print(f"\nTraining completed! Results saved to: {self.output_dir}")

    def _save_split_checkpoint(self, suffix: str = "") -> None:
        """
        Separately save Adapter and SAM3 weights.
        Useful for inference or modular deployment.
        """
        if not self.model.sam3_loader:
            return

        state_dict = self.model.state_dict()
        adapter_state = {}
        sam3_state = {}

        for k, v in state_dict.items():
            # Adapter parts
            if any(k.startswith(p) for p in ["adapter.", "seg_projector.", "seg_action_head."]):
                adapter_state[k] = v
            # SAM3 parts
            elif k.startswith("sam3_loader."):
                sam3_state[k] = v
        
        if adapter_state:
            path = self.output_dir / f"adapter_checkpoint{suffix}.pt"
            torch.save({"model_state_dict": adapter_state, "config": self.config}, path)
            print(f"  Split checkpoint saved: {path.name}")
            
        if sam3_state:
            path = self.output_dir / f"sam3_checkpoint{suffix}.pt"
            torch.save({"model_state_dict": sam3_state, "config": self.config}, path)
            print(f"  Split checkpoint saved: {path.name}")

    @torch.no_grad()
    def _extract_seg_features(self, dataloader: DataLoader, output_dir: Path) -> None:
        """Stage 1 训练完后, 自动提取 <SEG> hidden states 保存到 data/seg_features/。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.qwen_encoder.load_model()

        print(f"\n{'=' * 60}")
        print(f"提取 <SEG> features → {output_dir}")
        print(f"{'=' * 60}")

        count = 0
        dataset = dataloader.dataset

        for idx in tqdm(range(len(dataset)), desc="提取 <SEG>", leave=False):
            sample = dataset[idx]
            ann = dataset.annotations[idx]
            sample_id = ann.get("id", ann.get("scene_id", f"{dataset.split}_{idx:06d}"))

            out_path = output_dir / f"{sample_id}.pt"
            if out_path.exists():
                count += 1
                continue

            plane_img = sample["plane_image"]
            obj_img = sample["object_image"]
            text_prompt = sample["text_prompt"]

            seg_hidden, _ = self.model.qwen_encoder.generate_with_seg(
                text_prompt=text_prompt,
                images=[plane_img, obj_img],
                force_only=True,
                num_seg=self.config.get("model", {}).get("num_seg_tokens", 1),
            )

            seg_hidden = seg_hidden.squeeze(0).cpu().float()
            torch.save({"seg_hidden": seg_hidden, "sample_id": sample_id}, out_path)
            count += 1

        print(f"完成: {count} 个特征已保存到 {output_dir}")

    def _save_checkpoint(self, epoch: Any, val_loss: float, is_best: bool = False) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far (updates self.best_val_loss)
        """
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

        training_config = self.config.get("training", {})
        save_epoch = training_config.get("save_epoch", False)
        save_interval = training_config.get("save_interval", 10)

        # Save epoch checkpoint
        if epoch == "final":
            path = self.output_dir / "checkpoint_final.pt"
            torch.save(checkpoint, path)
        elif isinstance(epoch, int):
            # Save at specified intervals if save_epoch is enabled
            if save_epoch and (epoch + 1) % save_interval == 0:
                path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, path)

        # Save best checkpoint
        if is_best and training_config.get("save_best", True):
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            
            # Also save split weights for the best model in Stage 2
            if self.stage == "placement" and self.model.sam3_loader is not None and training_config.get("save_split_weights", True):
                self._save_split_checkpoint(suffix="_best")
