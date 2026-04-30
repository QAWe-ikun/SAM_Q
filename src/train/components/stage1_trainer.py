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

    def train(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train Stage 1 using SFTTrainer.
        """
        try:
            from trl import SFTTrainer  # type: ignore
            from transformers import TrainingArguments  # type: ignore
        except ImportError:
            raise ImportError("Please install trl: pip install trl>=0.8.0")

        training_config = self.config.get("training", {})
        lora_cfg = self.config.get("model", {}).get("qwen", {}).get("lora", {})

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
        grad_accum = training_config.get("gradient_accumulation_steps", 4)
        batch_size = training_config.get("batch_size", 2)
        num_epochs = training_config.get("num_epochs", 3)
        dataloader_num_workers = self.config.get("data", {}).get("num_workers",2)
        lr = self.config.get("optimizer", {}).get("lr", 1e-4)
        warmup_steps = self.config.get("scheduler", {}).get("warmup_epochs", 1)
        log_steps = training_config.get("log_interval", 10)
        use_bf16 = self.config.get("training", {}).get("bf16", False)
        use_fp16 = self.config.get("training", {}).get("fp16", True)

        sft_config = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            dataloader_num_workers = dataloader_num_workers,
            learning_rate=lr,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=log_steps,
            save_strategy="no",
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

        # Initialize SFTTrainer
        trainer = SFTTrainer(
            model=qwen_model,
            train_dataset=dataloader.dataset,
            data_collator=qwen_data_collator,
            args=sft_config,
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
