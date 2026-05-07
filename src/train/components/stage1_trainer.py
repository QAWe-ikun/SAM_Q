"""
Stage 1 Trainer
===============

SFT-based training for Qwen3-VL LoRA fine-tuning.
"""

import torch # type: ignore
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader  # type: ignore


class Stage1Trainer:
    """
    Stage 1: Fine-tune Qwen3-VL with LoRA using SFTTrainer.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
        output_dir: Path,
        device: str,
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.system_prompt = (
            "你是3D室内物体摆放助手。用户会提供：\n"
            "1. 场景平面参考图\n"
            "2. 目标物体的参考图\n"
            "3. 摆放指令\n\n"
            "请根据场景中的环境布局（墙壁、现有家具位置与朝向），"
            "推断目标物体应放置的位置、旋转角度和缩放比例。\n\n"
            "坐标系：上方=前，下方=后，左方=左，右方=右。\n"
            "旋转绕Y轴：正角度=顺时针（向右转），负角度=逆时针（向左转）。\n\n"
            "【输出格式】\n"
            "好的，我会将[物体]摆放在[具体位置]。"
            "物体在参考图中朝[前/后/左/右]，在场景中需要朝[前/后/左/右]摆放，且大小[偏大/偏小/正常]。"
            "所以要绕Y轴旋转[角度]°，缩放[比例]倍。"
            "综上所述，我会把物体放在<SEG>"
            "【强制顺序】你必须按以下顺序输出，缺一不可："
            "1. 位置描述"
            "2. 朝向判断"
            "3. 旋转角度和缩放比例"
            "4. 综上所述...<SEG>"
            "缺少任何一步都是错误输出。"
        )

    def train(self, dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Train Stage 1 using SFTTrainer.
        """
        print(f"[Train] val_dataloader: {val_dataloader is not None}, samples: {len(val_dataloader.dataset) if val_dataloader is not None else 0}")
        try:
            from trl import SFTTrainer  # type: ignore
            from transformers import TrainingArguments, TrainerCallback, EarlyStoppingCallback  # type: ignore
        except ImportError:
            raise ImportError("Please install trl: pip install trl>=0.8.0")

        training_config = self.config.get("training", {})
        lora_cfg = self.config.get("model", {}).get("qwen", {}).get("lora", {})

        # Ensure model is loaded with LoRA
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
        grad_accum = training_config.get("gradient_accumulation_steps", 4)
        batch_size = training_config.get("batch_size", 2)
        num_epochs = training_config.get("num_epochs", 3)
        dataloader_num_workers = self.config.get("data", {}).get("num_workers",2)
        lr = self.config.get("optimizer", {}).get("lr", 1e-4)
        warmup_steps = self.config.get("scheduler", {}).get("warmup_epochs", 1)
        log_steps = training_config.get("log_interval", 10)
        use_bf16 = self.config.get("training", {}).get("bf16", False)
        use_fp16 = self.config.get("training", {}).get("fp16", True)

        save_best = training_config.get("save_best", False)
        save_epoch = training_config.get("save_epoch", False)
        save_interval = training_config.get("save_interval", 100)
        save_total_limit = training_config.get("save_total_limit", 3)
        early_stopping = training_config.get("early_stopping", False)
        patience = training_config.get("patience", 3)

        eval_strategy = None
        eval_steps = None

        if early_stopping:
            eval_strategy = "steps"
            eval_steps = log_steps
            load_best_model_at_end = True
            metric_for_best_model = "eval_loss"
            greater_is_better = False
            save_strategy = "steps"
            save_steps = log_steps
        elif save_best:
            eval_strategy = "steps"
            eval_steps = log_steps
            save_strategy = "steps"
            save_steps = log_steps
            load_best_model_at_end = True
            metric_for_best_model = "eval_loss"
            greater_is_better = False
        elif save_epoch:
            save_strategy = "steps"
            steps_per_epoch = len(dataloader) // grad_accum
            save_steps = steps_per_epoch * save_interval
            load_best_model_at_end = False
            metric_for_best_model = None
            greater_is_better = None
        else:
            save_strategy = "no"
            save_steps = None
            load_best_model_at_end = False
            metric_for_best_model = None
            greater_is_better = None

        eval_batch_size = batch_size // 2 if batch_size > 1 else 1

        sft_config = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_num_workers = dataloader_num_workers,
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=log_steps,
            save_strategy=save_strategy,
            save_steps=save_steps if save_strategy != "no" else None,
            save_total_limit=save_total_limit,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            report_to="none",
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            weight_decay=self.config.get("optimizer", {}).get("weight_decay", 0.01),
            max_grad_norm=1.0,
            remove_unused_columns=False,
        )

        # Create data collator
        def qwen_data_collator(examples):
            return self._build_qwen_batch(examples, tokenizer)

        # 每 logging_steps 用数据集第一条数据检查 <SEG> / eos logit
        first_sample = dataloader.dataset[0]
        first_text_prompt = first_sample.get("text_prompt", "")
        first_images = first_sample.get("images", [])

        class TieWeightSyncCallback(TrainerCallback):
            """每步优化后同步 embed[<SEG>] 到 lm_head[<SEG>]，防止 tied weights 被 PEFT 破坏。"""
            def __init__(self, model, seg_id):
                self.model = model
                self.seg_id = seg_id

            def on_step_end(self, _args, state, control, **_kwargs):
                emb = self.model.get_input_embeddings().weight
                lm = self.model.get_output_embeddings().weight
                if emb.data_ptr() != lm.data_ptr():
                    with torch.no_grad():
                        lm[self.seg_id].copy_(emb[self.seg_id])
                return control

        class LogitCallback(TrainerCallback):
            def __init__(self, model, encoder, text_prompt, images, device):
                self.model = model
                self.encoder = encoder
                self.text_prompt = text_prompt
                self.images = images
                self.device = device
                self._seg_id = None
                self._eos_id = None
                self._test_inputs = None
                self._img_list = None

            def _setup(self):
                if self._seg_id is not None:
                    return
                tokenizer = self.encoder.processor.tokenizer
                self._seg_id = tokenizer.convert_tokens_to_ids('<SEG>')
                self._eos_id = tokenizer.eos_token_id
                messages, img_list = self.encoder._build_message(
                    text_prompt=self.text_prompt,
                    images=self.images,
                )
                test_text = self.encoder.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self._img_list = img_list if img_list else None
                self._test_inputs = self.encoder.processor(
                    text=[test_text],
                    images=self._img_list,
                    return_tensors="pt",
                ).to(self.device)

            def on_log(self, _args, _state, control, logs=None, **_kwargs):
                if logs is None:
                    return control
                self._setup()
                with torch.no_grad():
                    outputs = self.model(**self._test_inputs)
                last_logits = outputs.logits[0, -1]
                logs["seg_logit"] = last_logits[self._seg_id].item()
                logs["eos_logit"] = last_logits[self._eos_id].item()
                return control

        seg_id = tokenizer.convert_tokens_to_ids('<SEG>')

        # Build callbacks
        class EvalCacheCleanupCallback(TrainerCallback):
            def on_step_end(self, _args, state, control, **_kwargs):
                if state.global_step % 10 == 0:
                    torch.cuda.empty_cache()
                return control
            def on_evaluate(self, _args, _state, control, **_kwargs):
                torch.cuda.empty_cache()
                return control

        callbacks = [
            TieWeightSyncCallback(qwen_model, seg_id),
            LogitCallback(qwen_model, self.model.qwen_encoder, first_text_prompt, first_images, self.device),
            EvalCacheCleanupCallback(),
        ]
        if early_stopping:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=qwen_model,
            train_dataset=dataloader.dataset,
            data_collator=qwen_data_collator,
            args=sft_config,
            eval_dataset=val_dataloader.dataset if val_dataloader is not None else None,
            callbacks=callbacks,
        )

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

    def _build_qwen_batch(self, examples, tokenizer):
        """Custom collator for Qwen3-VL format."""
        texts = []
        images = []

        for ex in examples:
            text_prompt = ex.get("text_prompt", "")
            response = ex.get("response", "好的，我将为您放置物体。<SEG>")

            messages, img_list = self.model.qwen_encoder._build_message(
                text_prompt=text_prompt,
                images=ex.get("images", []),
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            ] + messages
            messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

            text = self.model.qwen_encoder.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            images.append(img_list if img_list else None)

        inputs = self.model.qwen_encoder.processor(
            text=texts,
            images=images if any(img is not None for img in images) else None,
            return_tensors="pt",
            padding=True,
        )

        # Create labels for LM loss
        labels = inputs["input_ids"].clone()
        for i, ex in enumerate(examples):
            response = ex.get("response", "")
            response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
            resp_len = len(response_tokens)
            real_len = inputs["attention_mask"][i].sum().item()

            if real_len > resp_len:
                labels[i, :real_len - resp_len] = -100
            labels[i, real_len:] = -100

        inputs["labels"] = labels
        return inputs

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Stage 1 validation with loss computation and sample generation."""
        self.model.eval()

        tokenizer = self.model.qwen_encoder.processor.tokenizer
        qwen_model = self.model.qwen_encoder.model

        # ===== 诊断代码：检查 LoRA 权重加载是否正确 =====
        print("=" * 60)
        seg_id = tokenizer.convert_tokens_to_ids('<SEG>')
        eos_id = tokenizer.eos_token_id

        # 1. 检查 embed_tokens 和 lm_head 是否共享同一块内存
        emb = qwen_model.get_input_embeddings().weight
        lm_head = qwen_model.get_output_embeddings().weight if hasattr(qwen_model, 'get_output_embeddings') else None
        tied = lm_head is not None and emb.data_ptr() == lm_head.data_ptr()
        print(f"1. embed_tokens == lm_head (tied): {tied}")
        if not tied:
            print("   ⚠️  embed 和 lm_head 未共享，强制同步 <SEG> 权重")
            lm_head[seg_id] = emb[seg_id].clone()

        # 2. 检查 tokenizer 词表
        print(f"2. <SEG> token id: {seg_id}, vocab size: {len(tokenizer)}")

        # 3. 检查 lm_head 中 <SEG> vs eos 的 norm
        if lm_head is not None:
            seg_norm = lm_head[seg_id].norm().item()
            eos_norm = lm_head[eos_id].norm().item()
            print(f"3. lm_head[<SEG>] norm: {seg_norm:.4f}, lm_head[eos] norm: {eos_norm:.4f}, diff: {abs(seg_norm - eos_norm):.4f}")
            if abs(seg_norm - eos_norm) < 1e-4:
                print("   ⚠️  <SEG> 和 eos 权重几乎一样，训练可能没生效")
            else:
                print("   ✅  <SEG> 权重已改变，训练有效")

        # 4. 检查生成参数
        print(f"5. model.config.eos_token_id: {qwen_model.config.eos_token_id}")
        print(f"   tokenizer.eos_token: {tokenizer.eos_token}")
        print("=" * 60)
        # ===== 诊断结束 =====

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

                    messages, image_list = self.model.qwen_encoder._build_message(
                        text_prompt=text_prompt,
                        images=sample_images,
                    )
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
                    ] + messages
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

                    input_ids = inputs["input_ids"]
                    labels = input_ids.clone()
                    response_tokens = tokenizer(response, add_special_tokens=False)["input_ids"]
                    resp_len = len(response_tokens)
                    labels[:, :-resp_len] = -100

                    outputs = qwen_model(**inputs, labels=labels)
                    loss = outputs.loss
                    if loss is not None:
                        batch_loss += loss.item()

                total_loss += batch_loss / batch_size
                num_batches += 1

                # Generate sample outputs for first batch
                if batch_idx == 0:
                    self._generate_samples(batch, qwen_model, tokenizer)

        return {"val_loss": total_loss / max(num_batches, 1)}

    def _generate_samples(self, batch, qwen_model, tokenizer):
        """Generate and print sample outputs for debugging."""
        print(f"\n{'='*60}")
        print(f"[Stage1 Validation Samples]")
        print(f"{'='*60}")

        num_gen_samples = min(2, len(batch["images"]))
        for i in range(num_gen_samples):
            sample_images = [img.to(self.device) for img in batch["images"][i]]
            text_prompt = batch["text_prompts"][i]
            response_gt = batch.get("responses", [None] * num_gen_samples)[i]

            messages, image_list = self.model.qwen_encoder._build_message(
                text_prompt=text_prompt,
                images=sample_images,
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            ] + messages

            text = self.model.qwen_encoder.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = self.model.qwen_encoder.processor(
                text=[text],
                images=image_list if image_list else None,
                return_tensors="pt",
                padding=True,
            ).to(self.device)

            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            assistant_marker = "assistant"
            if assistant_marker in generated_text:
                generated_text = generated_text.split(assistant_marker)[-1]

            print(f"Prompt:   {text_prompt[:60]}...")
            print(f"Generated: {generated_text.strip()}")
            print(f"Expected:  {response_gt.strip() if response_gt else 'N/A'}")
            print(f"{'-'*60}")
