"""
SAM-Q VLM 客户端模块

负责与 Qwen3-VL 模型交互，生成文本提示和回复。
支持 transformers 和 vLLM 两种推理后端。
"""

import torch  # type: ignore
import logging
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class VLMClient:
    """Qwen3-VL 客户端，支持 transformers 和 vLLM 两种推理后端"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        use_vllm: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.use_vllm = use_vllm
        self._model = None
        self._processor = None
        self._vllm_engine = None

    def load_model(self):
        """懒加载模型"""
        if self._model is not None or self._vllm_engine is not None:
            return

        model_path = Path(self.model_path)
        if not model_path.exists() or not (model_path / "config.json").exists():
            raise RuntimeError(f"Qwen3-VL 本地模型未找到: {model_path}")

        logger.info(f"加载 Qwen3-VL 模型: {model_path} (use_vllm={self.use_vllm})")

        if self.use_vllm:
            self._load_vllm_engine(model_path)
        else:
            self._load_transformers_model(model_path)

    def _load_vllm_engine(self, model_path: Path):
        """加载 vLLM 推理引擎"""
        try:
            from vllm import LLM  # type: ignore

            logger.info(f"使用 vLLM 引擎加载: {model_path}")
            self._vllm_engine = LLM(
                model=str(model_path),
                limit_mm_per_prompt={"image": 3},
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                trust_remote_code=True,
            )
            logger.info("vLLM 引擎加载成功")
        except ImportError:
            logger.warning("vLLM 未安装，回退到 transformers")
            self.use_vllm = False
            self._load_transformers_model(model_path)

    def _load_transformers_model(self, model_path: Path):
        """加载 transformers 模型"""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # type: ignore

        logger.info(f"使用 transformers 加载: {model_path}")

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_cache=True,
        )

        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn  # type: ignore
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self._model.config.use_cache = True
        self._model.eval()

        logger.info(f"模型加载完成: {self._model.device}")

    def generate_placement_description(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> str:
        """生成摆放位置描述"""
        if self.use_vllm:
            return self._vllm_generate_placement_description(
                original_image, plane_image, object_image, desc
            )
        else:
            return self._transformers_generate_placement_description(
                original_image, plane_image, object_image, desc
            )

    def _vllm_generate_placement_description(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> str:
        """vLLM 生成 placement description"""
        from vllm import SamplingParams  # type: ignore

        original_pil = Image.fromarray(original_image)
        plane_pil = Image.fromarray(plane_image)
        obj_pil = Image.fromarray(object_image)

        prompt_text = (
            "第一张图是包含所有物体的原始房间图，"
            "第二张图是移除了某个物体后的房间图，"
            "第三张图是被移除的物体的参考图。"
            f"请对比这三张图，用简短的中文描述被移除的物体{desc}"
            "原来放在什么位置，以及周围参照物的关系。"
            "以 '请你将[物体名称]摆放在' 开头。"
        )

        # vLLM 多模态输入格式
        vllm_input = {
            "prompt": f"<|image_1|>\n<|image_2|>\n<|image_3|>\n{prompt_text}",
            "multi_modal_data": {
                "image": [original_pil, plane_pil, obj_pil],
            },
        }

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )

        outputs = self._vllm_engine.generate([vllm_input], sampling_params)
        result = outputs[0].outputs[0].text.strip()
        logger.info(f"vLLM placement description: {result[:80]}...")
        return result

    def _transformers_generate_placement_description(
        self,
        original_image: np.ndarray,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        desc: str,
    ) -> str:
        """transformers 生成 placement description"""
        original_pil = Image.fromarray(original_image)
        plane_pil = Image.fromarray(plane_image)
        obj_pil = Image.fromarray(object_image)

        prompt_text = (
            "第一张图是包含所有物体的原始房间图，"
            "第二张图是移除了某个物体后的房间图，"
            "第三张图是被移除的物体的参考图。"
            f"请对比这三张图，用简短的中文描述被移除的物体{desc}"
            "原来放在什么位置，以及周围参照物的关系。"
            "以 '请你将[物体名称]摆放在' 开头。"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_pil},
                    {"type": "image", "image": plane_pil},
                    {"type": "image", "image": obj_pil},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[original_pil, plane_pil, obj_pil],
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        result = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        logger.info(f"transformers placement description: {result[:80]}...")
        return result

    def generate_response(
        self,
        text_prompt: str,
        rotation_6d: List[float],
        scale: float,
    ) -> str:
        """
        生成回复，包含旋转和缩放信息。

        Args:
            text_prompt: 放置指令（已包含 <image> 占位符）
            rotation_6d: 6D 旋转表示
            scale: 缩放比例

        Returns:
            包含旋转角度和缩放比例的回复，以 <SEG> 结尾
        """
        if self.use_vllm:
            return self._vllm_generate_response(text_prompt, rotation_6d, scale)
        else:
            return self._transformers_generate_response(text_prompt, rotation_6d, scale)

    def _vllm_generate_response(
        self,
        text_prompt: str,
        rotation_6d: List[float],
        scale: float,
    ) -> str:
        """vLLM 生成回复"""
        from vllm import SamplingParams  # type: ignore

        rot_y_deg = self.extract_rotation_y(rotation_6d)

        prompt = (
            f"你是一个物体放置助手。用户给出了放置指令，"
            f"请你用礼貌的语气回复，并在末尾加上<SEG>标记。"
            f"\n指令：{text_prompt}"
            f"\n旋转角度：{rot_y_deg:.1f}°（绕Y轴）"
            f"\n缩放比例：{scale:.2f}"
            f"\n请用'好的，我会...'开头回复，说明放置位置、"
            f"旋转角度和缩放比例，并在句末加上<SEG>。"
        )

        # 从 prompt 中提取图片占位符数量
        image_count = prompt.count("<image>")

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

        outputs = self._vllm_engine.generate(
            [{"prompt": prompt}], sampling_params
        )
        result = outputs[0].outputs[0].text.strip()

        if "<SEG>" not in result:
            result += "<SEG>"

        logger.debug(f"vLLM response: {result[:80]}...")
        return result

    def _transformers_generate_response(
        self,
        text_prompt: str,
        rotation_6d: List[float],
        scale: float,
    ) -> str:
        """transformers 生成回复"""
        rot_y_deg = self.extract_rotation_y(rotation_6d)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"你是一个物体放置助手。用户给出了放置指令，"
                            f"请你用礼貌的语气回复，并在末尾加上<SEG>标记。"
                            f"\n指令：{text_prompt}"
                            f"\n旋转角度：{rot_y_deg:.1f}°（绕Y轴）"
                            f"\n缩放比例：{scale:.2f}"
                            f"\n请用'好的，我会...'开头回复，说明放置位置、"
                            f"旋转角度和缩放比例，并在句末加上<SEG>。"
                        )
                    }
                ]
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
        ).to(self._model.device)

        from transformers import GenerationConfig  # type: ignore
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        result = self._processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        if "<SEG>" not in result:
            result += "<SEG>"

        logger.debug(f"transformers response: {result[:80]}...")
        return result
