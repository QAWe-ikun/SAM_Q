"""
Qwen3-VL Encoder for SAM3 Text Encoder Replacement

This module wraps Qwen3-VL to replace SAM3's Text Encoder,
enabling multimodal input (object image + text prompt).
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
        model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
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
        
    def load_model(self):
        """Lazy load the Qwen3-VL model and processor."""
        if self.model is not None:
            return
            
        try:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            )
            self.model.eval()
            
            # Set output dimension based on Qwen3-VL hidden size
            self._output_dim = self.model.config.hidden_size
            
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
        
        # Process inputs
        inputs = self.processor(
            text=messages,
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


class Qwen3VLEncoderWithProjection(nn.Module):
    """
    Qwen3-VL Encoder with projection layer for SAM3 compatibility.
    
    Projects Qwen3-VL embeddings to SAM3's expected input dimension.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
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
