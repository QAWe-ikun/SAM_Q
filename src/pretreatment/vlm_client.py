"""
SAM-Q VLM 客户端模块

负责与 Qwen3-VL 模型交互，生成文本提示和回复。
"""

import torch # type: ignore
import logging
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class VLMClient:
    """Qwen3-VL 客户端"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None

    def load_model(self):
        """懒加载模型"""
        if self._model is not None:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # type: ignore

        model_path = Path(self.model_path)
        if not model_path.exists() or not (model_path / "config.json").exists():
            raise RuntimeError(f"Qwen3-VL 本地模型未找到: {model_path}")

        logger.info(f"加载 Qwen3-VL 模型: {model_path}")

        # 加载 processor
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_cache=True,
        )

        # 确定 attention 实现
        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn  # type: ignore
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

        # 加载模型
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self._model.config.use_cache = True
        self._model.eval()

        logger.info(f"模型加载完成: {self._model.device}")

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.load_model()
        return self._processor

    def generate_placement_description(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> str:
        """
        生成摆放位置描述。

        Args:
            original_image: 原始房间图
            plane_image: 剔除后房间图
            object_image: 物体参考图
            desc: 物体描述

        Returns:
            位置描述文本
        """
        original_pil = Image.fromarray(original_image)
        plane_pil = Image.fromarray(plane_image)
        obj_pil = Image.fromarray(object_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_pil},
                    {"type": "image", "image": plane_pil},
                    {"type": "image", "image": obj_pil},
                    {
                        "type": "text",
                        "text": (
                            "第一张图是包含所有物体的原始房间图，"
                            "第二张图是移除了某个物体后的房间图，"
                            "第三张图是被移除的物体的参考图。"
                            f"请对比这三张图，用简短的中文描述被移除的物体{desc}"
                            "原来放在什么位置，以及周围参照物的关系。"
                            "以 '请你将[物体名称]摆放在' 开头。"
                        )
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[original_pil, plane_pil, obj_pil],
            return_tensors="pt",
        ).to(self.model.device)

        from transformers import GenerationConfig  # type: ignore
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.decode(
            outputs[0, input_len:], skip_special_tokens=True
        ).strip()

        return response

    @staticmethod
    def extract_rotation_y(rotation_6d: List[float]) -> float:
        """从 6D 旋转表示中提取绕 Y 轴的旋转角度（度数）"""
        import math

        r11, r21, r31, r12, r22, r32 = rotation_6d

        # 计算第三列（叉乘）
        r33 = r11 * r22 - r21 * r12

        # 提取绕 Y 轴的旋转角度
        rot_y = math.atan2(-r31, r33)
        rot_y_deg = math.degrees(rot_y)

        # 规范化到 [-180, 180]
        if rot_y_deg > 180:
            rot_y_deg -= 360
        elif rot_y_deg < -180:
            rot_y_deg += 360

        return rot_y_deg

    def generate_response(
        self,
        text_prompt: str,
        rotation_6d: List[float],
        scale: float,
    ) -> str:
        """
        生成 response。

        Args:
            text_prompt: 放置指令
            rotation_6d: 6D 旋转表示
            scale: 缩放比例

        Returns:
            包含旋转和缩放信息的回复
        """
        rot_y_deg = self.extract_rotation_y(rotation_6d)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是一个物体放置助手。用户给出了放置指令，"
                            "请你用礼貌的语气回复，并在末尾加上<SEG>标记。"
                            f"\n指令：{text_prompt}"
                            f"\n旋转角度：{rot_y_deg:.1f}°（绕Y轴）"
                            f"\n缩放比例：{scale:.2f}"
                            "\n请用'好的，我会...'开头回复，说明放置位置、"
                            "旋转角度和缩放比例，并在句末加上<SEG>。"
                        )
                    }
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
        ).to(self.model.device)

        from transformers import GenerationConfig  # type: ignore
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        response = self.processor.decode(
            outputs[0, input_len:], skip_special_tokens=True
        ).strip()

        if "<SEG>" not in response:
            response += "<SEG>"

        return response
