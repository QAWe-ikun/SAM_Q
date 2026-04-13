"""
Qwen3-VL Encoder for SAM3 Text Encoder Replacement

This module wraps Qwen3-VL to replace SAM3's Text Encoder,
enabling multimodal input (object image + text prompt).

Supports two modes:
  - Encoding mode: standard forward pass, returns full sequence hidden states
  - SEG token mode: generates/forces a [SEG] token, returns its hidden state
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from PIL import Image


class Qwen3VLEncoder(nn.Module):
    """
    Qwen3-VL Encoder that replaces SAM3's Text Encoder.
    
    Takes object image and text prompt as input,
    outputs embeddings compatible with SAM3 Detector.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        # Load Qwen3-VL model (lazy loading to avoid import errors during development)
        self.model = None
        self.processor = None
        
        # Placeholder for output dimension (will be set after model loading)
        self._output_dim = None

        # [SEG] token ID (set after model loading)
        self.seg_token_id = None
        
    def load_model(self):
        """Lazy load the Qwen3-VL model and processor."""
        if self.model is not None:
            return
            
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)

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
                attn_implementation=attn_impl,
            )
            self.model.eval()
            
            # Set output dimension based on Qwen3-VL hidden size
            cfg = self.model.config
            self._output_dim = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

            # Register [SEG] special token for SA2VA-style bridging
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["[SEG]"]}
            )
            self.model.resize_token_embeddings(len(self.processor.tokenizer))
            self.seg_token_id = self.processor.tokenizer.convert_tokens_to_ids("[SEG]")
            
        except ImportError as e:
            raise ImportError(
                "Please install required packages: "
                "pip install transformers qwen-vl-utils"
            ) from e
    
    @property
    def output_dim(self) -> int:
        """Get the output embedding dimension."""
        if self._output_dim is None:
            self.load_model()
        return self._output_dim
    
    def forward(
        self,
        object_image: Optional[Image.Image] = None,
        text_prompt: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through Qwen3-VL.
        
        Args:
            object_image: PIL Image of the object (top-down view)
            text_prompt: Text description for placement instruction
            **kwargs: Additional arguments for model
            
        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        self.load_model()

        # Build conversation for Qwen3-VL
        messages = self._build_message(object_image, text_prompt)

        # Apply chat template to convert messages → text string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[object_image] if object_image else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Get embeddings from the model
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
        object_image: Optional[Image.Image],
        text_prompt: Optional[str],
    ) -> list:
        """
        Build conversation message for Qwen3-VL.
        
        Args:
            object_image: Object top-down view image
            text_prompt: Placement text prompt
            
        Returns:
            messages: Formatted message list for Qwen3-VL
        """
        default_prompt = "Describe the object and suggest good placement positions."
        
        if object_image is not None:
            content = [
                {"type": "image"},
                {"type": "text", "text": text_prompt or default_prompt},
            ]
        else:
            content = [
                {"type": "text", "text": text_prompt or default_prompt},
            ]
        
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        
        return messages
    
    def encode_text_only(self, text: str) -> torch.Tensor:
        """Encode text-only prompt without image."""
        return self.forward(text_prompt=text)
    
    def encode_multimodal(
        self,
        image: Image.Image,
        text: str,
    ) -> torch.Tensor:
        """Encode combined image and text input."""
        return self.forward(object_image=image, text_prompt=text)

    def generate_with_seg(
        self,
        object_image: Optional[Image.Image] = None,
        text_prompt: Optional[str] = None,
        max_new_tokens: int = 128,
        force_only: bool = True,
    ) -> Tuple[torch.Tensor, bool]:
        """
        Extract [SEG] token hidden state for SAM3 prompt generation.

        Two modes:
          - force_only=True (training): append [SEG] to input, single forward
            pass, extract hidden state at [SEG] position. Fast, no generation.
          - force_only=False (inference): generate autoregressively, check if
            [SEG] was produced. If not, force-append it. Then extract hidden state.

        Args:
            object_image: PIL Image of the object
            text_prompt: Text placement instruction
            max_new_tokens: Max tokens to generate (only when force_only=False)
            force_only: Skip generation, directly force [SEG]

        Returns:
            seg_hidden_state: [B, hidden_dim] tensor
            was_natural: True if [SEG] was generated by the model (always False when force_only=True)
        """
        self.load_model()

        messages = self._build_message(object_image, text_prompt)

        # Tokenize input
        inputs = self.processor(
            text=messages,
            images=[object_image] if object_image else None,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        if force_only:
            return self._force_seg_forward(inputs)

        return self._generate_and_extract_seg(inputs, max_new_tokens)

    def _force_seg_forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, bool]:
        """Append [SEG] to input and extract its hidden state in one forward pass."""
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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # [SEG] is the last token
        seg_hidden = outputs.hidden_states[-1][:, -1, :]
        return seg_hidden, False

    def _generate_and_extract_seg(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, bool]:
        """Generate autoregressively, then extract [SEG] hidden state."""
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Check if [SEG] was generated
        generated_ids = generated[:, inputs["input_ids"].size(1):]
        seg_positions = (generated_ids == self.seg_token_id).nonzero(as_tuple=False)
        was_natural = len(seg_positions) > 0

        if not was_natural:
            # Force-append [SEG]
            seg_ids = torch.full(
                (generated.size(0), 1),
                self.seg_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
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
            # Extract hidden state at first [SEG] position per batch item
            batch_size = generated.size(0)
            seg_hidden_list = []
            for b in range(batch_size):
                seg_mask = (generated[b] == self.seg_token_id)
                seg_idx = seg_mask.nonzero(as_tuple=False)[0, 0]
                seg_hidden_list.append(hidden_states[b, seg_idx, :])
            seg_hidden = torch.stack(seg_hidden_list, dim=0)
        else:
            # [SEG] is the last token
            seg_hidden = hidden_states[:, -1, :]

        return seg_hidden, was_natural


class Qwen3VLEncoderWithProjection(nn.Module):
    """
    Qwen3-VL Encoder with projection layer for SAM3 compatibility.
    
    Projects Qwen3-VL embeddings to SAM3's expected input dimension.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        sam3_input_dim: int = 256,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        
        self.encoder = Qwen3VLEncoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        
        # Projection layer to match SAM3 input dimension
        self.projection = nn.Linear(
            self.encoder.output_dim,
            sam3_input_dim,
        )
        
    def forward(
        self,
        object_image: Optional[Image.Image] = None,
        text_prompt: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Forward pass with projection.
        
        Returns:
            projected_embeddings: Tensor of shape (batch_size, seq_len, sam3_input_dim)
        """
        embeddings = self.encoder(object_image, text_prompt)
        projected = self.projection(embeddings)
        return projected
