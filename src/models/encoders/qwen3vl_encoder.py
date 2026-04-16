"""
Qwen3-VL Encoder for SAM3 Text Encoder Replacement

This module wraps Qwen3-VL to replace SAM3's Text Encoder,
enabling multimodal input (object image + text prompt).

Supports three modes:
  - Encoding mode: standard forward pass, returns full sequence hidden states
  - SEG token mode (single): generates/forces a <SEG> token, returns its hidden state
  - Multi-SEG token mode (SA2VA-style): generates/forces multiple [SEG0]~[SEG63] tokens
  - Fine-tuning mode: LoRA/QLoRA enabled, returns logits + hidden states for joint training
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image


class Qwen3VLEncoder(nn.Module):
    """
    Qwen3-VL Encoder that replaces SAM3's Text Encoder.

    Takes object image and text prompt as input,
    outputs embeddings compatible with SAM3 Detector.

    Features:
        - Single <SEG> token mode (backward compatible)
        - Multi [SEG0]~[SEG63] token mode (SA2VA-style)
        - LoRA/QLoRA fine-tuning support
        - Training/inference mode switching
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        num_seg_tokens: int = 1,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.num_seg_tokens = num_seg_tokens

        # Load Qwen3-VL model (lazy loading to avoid import errors during development)
        self.model = None
        self.processor = None

        # Placeholder for output dimension (will be set after model loading)
        self._output_dim = None

        # <SEG> token IDs (set after model loading)
        self.seg_token_id: Optional[int] = None  # Single <SEG> (backward compatible)
        self.seg_token_ids: List[int] = []       # Multi [SEG0]~[SEG63]

        # Fine-tuning state
        self.training_mode: bool = False
        self.lora_config: Optional[Dict] = None
        
    def load_model(self, use_cache: bool = True):
        """Lazy load the Qwen3-VL model and processor."""
        if self.model is not None:
            return

        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name, use_cache=use_cache)

            # Try flash_attention_2, fall back to sdpa/eager
            attn_impl = "eager"
            if self.device == "cuda":
                try:
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                except ImportError:
                    attn_impl = "sdpa"  # PyTorch native, nearly same speed

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                attn_implementation=attn_impl
            )
            
            # Set use_cache for generation (important for SEG token generation)
            self.model.config.use_cache = use_cache
            
            # Set output dimension based on Qwen3-VL hidden size
            cfg = self.model.config
            self._output_dim = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

            # Register <SEG> token(s) for SA2VA-style bridging
            # num_seg_tokens=1 → registers <SEG> (single mode)
            # num_seg_tokens>1 → registers [SEG0]~[SEG{n-1}] (multi mode)
            if self.num_seg_tokens == 1:
                seg_tokens = ["<SEG>"]
            else:
                seg_tokens = [f"[SEG{i}]" for i in range(self.num_seg_tokens)]

            # Register SEG tokens only
            self.processor.tokenizer.add_tokens(seg_tokens)
            self.model.resize_token_embeddings(len(self.processor.tokenizer))

            self.seg_token_ids = [
                self.processor.tokenizer.convert_tokens_to_ids(t) for t in seg_tokens
            ]
            self.seg_token_id = self.seg_token_ids[0]

            # For VLA: <SEG> serves dual purpose (segmentation + action)
            # Use seg_token_id as the action token ID
            self.exec_token_id = self.seg_token_id

            # Default to eval mode (training mode is enabled by enable_finetuning)
            self.model.eval()

            print(f"[Qwen3VLEncoder] Registered {len(seg_tokens)} SEG token(s): {seg_tokens}")
            print(f"[Qwen3VLEncoder] <SEG> token ID (also used as [EXEC]): {self.seg_token_id}")

        except ImportError as e:
            raise ImportError(
                "Please install required packages: "
                "pip install transformers qwen-vl-utils"
            ) from e

    def enable_finetuning(
        self,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        use_qlora: bool = False,
        lora_bias: str = "none",
    ):
        """
        Enable LoRA/QLoRA fine-tuning mode.

        Freezes vision encoder and SAM3-related parts, applies LoRA to
        language model layers only.

        Args:
            lora_r: LoRA rank (controls new matrix width)
            lora_alpha: Scaling factor (typically 2x lora_r)
            lora_dropout: Dropout for regularization
            target_modules: List of module names to apply LoRA.
                           Default: attention + MLP projections
            use_qlora: Whether to use 4-bit quantization (QLoRA)
            lora_bias: Bias handling ("none", "all", "lora_only")

        Returns:
            trainable_params: Number of trainable parameters
            trainable_pct: Percentage of trainable parameters
        """
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "Please install peft: pip install peft"
            ) from e

        self.load_model(use_cache=False)  # Ensure model is loaded before applying LoRA

        # Optional: QLoRA 4-bit quantization
        if use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)

        # Default target modules (attention + MLP)
        if target_modules is None:
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention
                "gate_proj", "up_proj", "down_proj",      # MLP
            ]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules,
            "use_qlora": use_qlora,
        }

        # Enable gradient checkpointing (reduces memory by ~40%)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        # For transformer-based models
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "gradient_checkpointing_enable"):
            self.model.base_model.gradient_checkpointing_enable()

        # Print trainable parameters
        trainable_params, all_param = self.model.get_nb_trainable_parameters()
        trainable_pct = 100 * trainable_params / all_param
        print(f"[Qwen3VLEncoder] LoRA enabled:")
        print(f"  Trainable params: {trainable_params:,} / {all_param:,} ({trainable_pct:.2f}%)")
        print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}")
        print(f"  Target modules: {target_modules}")

        # Switch to training mode
        self.training_mode = True
        self.model.train()

        # Ensure <SEG> token embeddings are trainable
        # (they should be by default since they were just added)
        return trainable_params, trainable_pct

    def disable_finetuning(self):
        """Disable fine-tuning mode, switch back to inference mode."""
        self.training_mode = False
        if self.model is not None:
            self.model.eval()

    @property
    def output_dim(self) -> int:
        """Get the output embedding dimension."""
        if self._output_dim is None:
            self.load_model()
        return self._output_dim
    
    def forward(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass through Qwen3-VL.

        Args:
            text_prompt: Text with optional <image> placeholders
            images: List of PIL images in order
            labels: Label token IDs for computing loss (training mode only)
            **kwargs: Additional arguments for model

        Returns:
            If training_mode or labels provided:
                dict with 'logits', 'hidden_states', and optionally 'loss'
            Otherwise (inference mode):
                embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        self.load_model()

        # Build conversation for Qwen3-VL
        messages, image_list = self._build_message(
            text_prompt=text_prompt or "",
            images=images,
        )

        # Apply chat template to convert messages → text string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process inputs
        # image_list contains PIL images in the order they appear in messages
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if labels is not None:
            inputs["labels"] = labels

        # Training mode: return logits + hidden states (with gradients)
        if self.training_mode or labels is not None:
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )
            return {
                "logits": outputs.logits,
                "hidden_states": outputs.hidden_states[-1],
                "loss": outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None,
            }

        # Inference mode: return embeddings only (no gradients)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
            )

        # Extract last hidden state
        embeddings = outputs.hidden_states[-1]
        return embeddings

    def _build_message(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
    ) -> Tuple[list, List[Image.Image]]:
        """
        Build conversation message for Qwen3-VL with flexible image placement.

        Images are embedded at <image> placeholders in the text.
        If no <image> placeholders found, images are prepended to the text.

        Args:
            text_prompt: Text with optional <image> placeholders
            images: List of PIL images in the order they should appear

        Returns:
            messages: Formatted message list for Qwen3-VL
            image_list: List of PIL images in the order they appear

        Examples:
            >>> # No placeholder: images prepended to text
            >>> _build_message("Describe this", images=[obj])
            >>> # → [image, "Describe this"]

            >>> # With placeholders: images embedded at positions
            >>> _build_message("房间<image>，放椅子<image>在角落", images=[room, chair])
            >>> # → ["房间", image(room), "，放椅子", image(chair), "在角落"]
        """
        if images:
            # 确保所有 image 都是 PIL.Image（训练时可能传入 tensor）
            pil_images = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    t = img.detach().cpu()
                    if t.dim() == 4:
                        t = t[0]  # [B,C,H,W] → [C,H,W]
                    if t.dtype != torch.uint8:
                        t = (t.clamp(0, 1) * 255).to(torch.uint8)
                    # [C,H,W] → [H,W,C]
                    t = t.permute(1, 2, 0).numpy()
                    pil_images.append(Image.fromarray(t))
                else:
                    pil_images.append(img)
            image_list = pil_images
        else:
            image_list = []

        # Parse text for <image> placeholders
        if "<image>" in text_prompt and image_list:
            parts = text_prompt.split("<image>")
            content = []
            img_idx = 0

            for i, part in enumerate(parts):
                if part.strip():
                    content.append({"type": "text", "text": part.strip()})
                # After each part (except last), insert image if available
                if i < len(parts) - 1 and img_idx < len(image_list):
                    content.append({"type": "image"})
                    img_idx += 1

            # Handle case where text ends with <image>
            if text_prompt.endswith("<image>") and img_idx < len(image_list):
                content.append({"type": "image"})
                img_idx += 1
        else:
            # No placeholders: prepend all images
            content = []
            for _ in image_list:
                content.append({"type": "image"})
            if text_prompt.strip():
                content.append({"type": "text", "text": text_prompt.strip()})

        messages = [{"role": "user", "content": content}]
        return messages, image_list
    
    def encode_text_only(self, text: str) -> torch.Tensor:
        """Encode text-only prompt without image."""
        return self.forward(text_prompt=text)

    def encode_multimodal(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
    ) -> torch.Tensor:
        """Encode combined image(s) and text input."""
        return self.forward(text_prompt=text, images=images)

    def generate_with_seg(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        max_new_tokens: int = 128,
        force_only: bool = True,
        num_seg: int = 1,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Extract hidden states up to and including <SEG> token(s).

        Returns all token hidden states from the start of the sequence
        up to (and including) the <SEG> token(s), so that the adapter
        can attend to the full context including any "reasoning" tokens.

        Returns:
            seg_hidden_states: [B, seg_pos+1, hidden_dim] — all tokens up to <SEG>
            was_natural: True if <SEG> was generated by the model
        """
        self.load_model()

        messages, image_list = self._build_message(
            text_prompt=text_prompt,
            images=images,
        )

        # Convert messages to text string via chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize input
        inputs = self.processor(
            text=[text],
            images=image_list if image_list else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if force_only:
            return self._force_seg_forward(inputs, num_seg)

        return self._generate_and_extract_seg(inputs, max_new_tokens, num_seg)

    def _force_seg_forward(
        self,
        inputs: Dict[str, torch.Tensor],
        num_seg: int = 1,
    ) -> Tuple[torch.Tensor, bool]:
        """Append single <SEG> to input and extract its hidden state."""
        input_ids = inputs["input_ids"]
        seg_ids = torch.full(
            (input_ids.size(0), 1),
            self.seg_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([input_ids, seg_ids], dim=1)

        # Extend attention mask
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            seg_mask = torch.ones(
                attention_mask.size(0), 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, seg_mask], dim=1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # <SEG> is the last token
        seg_hidden = outputs.hidden_states[-1][:, -1, :]
        return seg_hidden, False

    def _force_multi_seg_forward(
        self,
        inputs: Dict[str, torch.Tensor],
        num_seg: int,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Append multiple [SEG0]~[SEG{n-1}] tokens and extract their hidden states.

        Returns:
            seg_hidden_states: [B, num_seg, hidden_dim]
            was_natural: False (always forced)
        """
        input_ids = inputs["input_ids"]
        batch_size = input_ids.size(0)

        # Create SEG token IDs: [SEG0], [SEG1], ..., [SEG{num_seg-1}]
        seg_ids_to_append = self.seg_token_ids[:num_seg]
        seg_tensor = torch.tensor(
            seg_ids_to_append,
            dtype=input_ids.dtype,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, -1)  # [B, num_seg]

        input_ids = torch.cat([input_ids, seg_tensor], dim=1)

        # Extend attention mask
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            seg_mask = torch.ones(
                batch_size, num_seg,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, seg_mask], dim=1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # Extract hidden states for the last num_seg tokens
        hidden_states = outputs.hidden_states[-1]
        seg_hidden = hidden_states[:, -num_seg:, :]  # [B, num_seg, hidden_dim]

        return seg_hidden, False

    def _generate_and_extract_seg(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        num_seg: int = 1,
    ) -> Tuple[torch.Tensor, bool]:
        """Generate autoregressively, then extract <SEG> hidden state(s)."""
        with torch.no_grad():
            # Clear invalid generation kwargs to suppress warning
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        # Check if any SEG tokens were generated
        generated_ids = generated[:, inputs["input_ids"].size(1):]

        # For multi-SEG: check if any of the SEG token IDs appear
        was_natural = False
        if num_seg == 1:
            seg_positions = (generated_ids == self.seg_token_id).nonzero(as_tuple=False)
            was_natural = len(seg_positions) > 0
        else:
            # Check for any of the SEG0~SEG{n-1} tokens
            for seg_id in self.seg_token_ids[:num_seg]:
                seg_positions = (generated_ids == seg_id).nonzero(as_tuple=False)
                if len(seg_positions) > 0:
                    was_natural = True
                    break

        if not was_natural:
            # Force-append SEG token(s)
            if num_seg == 1:
                seg_ids = torch.full(
                    (generated.size(0), 1),
                    self.seg_token_id,
                    dtype=generated.dtype,
                    device=generated.device,
                )
            else:
                seg_ids_to_append = self.seg_token_ids[:num_seg]
                seg_ids = torch.tensor(
                    seg_ids_to_append,
                    dtype=generated.dtype,
                    device=generated.device,
                ).unsqueeze(0).expand(generated.size(0), -1)

            generated = torch.cat([generated, seg_ids], dim=1)

        # Forward pass on full sequence to get hidden states
        attention_mask = torch.ones_like(generated)
        with torch.no_grad():
            outputs = self.model(
                input_ids=generated,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]

        if was_natural:
            # Extract hidden states at SEG positions
            batch_size = generated.size(0)
            seg_hidden_list = []
            for b in range(batch_size):
                seg_indices = []
                for seg_id in self.seg_token_ids[:num_seg] if num_seg > 1 else [self.seg_token_id]:
                    seg_mask = (generated[b] == seg_id)
                    seg_idx = seg_mask.nonzero(as_tuple=False)
                    if len(seg_idx) > 0:
                        seg_indices.append(seg_idx[0, 0])

                if not seg_indices:
                    # Fallback: use last token
                    seg_indices = [generated.size(1) - 1]

                if num_seg == 1:
                    seg_hidden_list.append(hidden_states[b, seg_indices[0], :])
                else:
                    # Pad or truncate to num_seg
                    while len(seg_indices) < num_seg:
                        seg_indices.append(seg_indices[-1])
                    seg_hidden_list.append(
                        hidden_states[b, torch.tensor(seg_indices[:num_seg], device=hidden_states.device), :]
                    )

            if num_seg == 1:
                seg_hidden = torch.stack(seg_hidden_list, dim=0)  # [B, hidden_dim]
            else:
                seg_hidden = torch.stack(seg_hidden_list, dim=0)  # [B, num_seg, hidden_dim]
        else:
            # SEG token(s) are at the end
            if num_seg == 1:
                seg_hidden = hidden_states[:, -1, :]
            else:
                seg_hidden = hidden_states[:, -num_seg:, :]

        return seg_hidden, was_natural

    def load_lora_adapter(self, lora_path: str):
        """
        Load a pre-trained LoRA adapter into the model.

        This is used to load Stage 1 fine-tuned weights for inference.

        Args:
            lora_path: Path to the LoRA adapter directory
        """
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "Please install peft: pip install peft"
            ) from e

        self.load_model(use_cache=True)  # Inference mode

        # Load the LoRA adapter on top of the base model
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        self.training_mode = False

        print(f"[Qwen3VLEncoder] Loaded LoRA adapter from {lora_path}")

    def merge_lora(self):
        """
        Merge LoRA weights into the base model and return merged model.

        This is useful for inference speedup after fine-tuning.
        """
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError(
                "Please install peft: pip install peft"
            ) from e

        if not isinstance(self.model, PeftModel):
            print("[Qwen3VLEncoder] Model is not a PeftModel, skipping merge.")
            return

        self.model = self.model.merge_and_unload()
        print("[Qwen3VLEncoder] LoRA weights merged into base model.")


class Qwen3VLEncoderWithProjection(nn.Module):
    """
    Qwen3-VL Encoder with projection layer for SAM3 compatibility.

    Projects Qwen3-VL embeddings (or SEG hidden states) to SAM3's expected
    input dimension.

    Supports:
        - Single SEG token: [B, hidden_dim] → [B, sam3_input_dim]
        - Multi SEG tokens: [B, num_seg, hidden_dim] → [B, num_seg, sam3_input_dim]
        - Fine-tuning mode with LoRA
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        sam3_input_dim: int = 256,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        num_seg_tokens: int = 1,
    ):
        super().__init__()

        self.encoder = Qwen3VLEncoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
            num_seg_tokens=num_seg_tokens,
        )

        # Projection layer to match SAM3 input dimension
        self.projection = nn.Linear(
            self.encoder.output_dim,
            sam3_input_dim,
        )

    def enable_finetuning(self, **kwargs):
        """Enable LoRA fine-tuning on the underlying encoder."""
        return self.encoder.enable_finetuning(**kwargs)

    def disable_finetuning(self):
        """Disable fine-tuning mode."""
        self.encoder.disable_finetuning()

    def forward(
        self,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[torch.Tensor] = None,
        num_seg: int = 1,
        force_only: bool = True,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass with projection.

        Args:
            text_prompt: Text with optional <image> placeholders
            images: List of PIL images in order
            labels: Labels for training mode
            num_seg: Number of SEG tokens to use
            force_only: Force SEG tokens (training) or generate (inference)

        Returns:
            If training mode:
                dict with 'logits', 'hidden_states', 'projected_seg', 'loss'
            Otherwise:
                projected_seg: [B, sam3_input_dim] or [B, num_seg, sam3_input_dim]
        """
        # Training mode: get logits + hidden states
        if self.encoder.training_mode or labels is not None:
            encoder_output = self.encoder(
                text_prompt=text_prompt,
                images=images,
                labels=labels,
            )
            # Project SEG hidden states
            seg_hidden = encoder_output["hidden_states"]
            projected = self.projection(seg_hidden)
            encoder_output["projected_hidden_states"] = projected
            return encoder_output

        # Inference mode: extract and project SEG hidden states
        seg_hidden, was_natural = self.encoder.generate_with_seg(
            text_prompt=text_prompt,
            images=images,
            force_only=force_only,
            num_seg=num_seg,
        )
        projected = self.projection(seg_hidden)
        return projected, was_natural
