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

import os
import warnings

# 1. 环境变量
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"

# 2. 屏蔽所有相关警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*Fast.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")

# 3. 日志级别
import logging
logging.getLogger("vllm").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
            # WSL 环境需要设置 spawn 多进程启动方式
            import os
            import multiprocessing as mp
            mp.set_start_method("spawn", force=True)
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            # 禁用 flash_attn，避免 CUDA 符号冲突
            os.environ["VLLM_ATTENTION_BACKEND"] = "TORCH_SDPA"
            
            # 抑制 vLLM 内部的 INFO 日志
            import logging
            vllm_logger = logging.getLogger("vllm")
            vllm_logger.setLevel(logging.ERROR)

            from vllm import LLM  # type: ignore

            logger.info(f"使用 vLLM 引擎加载: {model_path}")
            self._vllm_engine = LLM(
                model=str(model_path),
                limit_mm_per_prompt={"image": 3},
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                swap_space=8,  # 启用 CPU 交换空间
                trust_remote_code=True,
                disable_log_stats=True,
                enforce_eager=True,
                disable_custom_all_reduce=True,
            )

            # 也加载 processor 用于生成 chat template（vLLM 需要正确格式的 prompt）
            from transformers import AutoProcessor  # type: ignore
            self._processor = AutoProcessor.from_pretrained(
                str(model_path),
                use_fast=True,
            )

            logger.info("vLLM 引擎加载成功")
        except ImportError:
            logger.warning("vLLM 未安装，回退到 transformers")
            self.use_vllm = False
            self._load_transformers_model(model_path)
        except Exception as e:
            logger.warning(f"vLLM 加载失败: {e}，回退到 transformers")
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

    def generate_placement_description_batch(
        self,
        original_images: List[np.ndarray],
        plane_images: List[np.ndarray],
        object_images: List[np.ndarray],
        descs: List[str],
    ) -> List[str]:
        """批量生成摆放位置描述"""
        if self.use_vllm:
            return self._vllm_generate_placement_description_batch(
                original_images, plane_images, object_images, descs
            )
        else:
            # transformers 不支持多模态批量，逐个生成
            results = []
            for i in range(len(descs)):
                result = self._transformers_generate_placement_description(
                    original_images[i], plane_images[i], object_images[i], descs[i]
                )
                results.append(result)
            return results

    def _vllm_generate_placement_description_batch(
        self,
        original_images: List[np.ndarray],
        plane_images: List[np.ndarray],
        object_images: List[np.ndarray],
        descs: List[str],
    ) -> List[str]:
        """vLLM 批量生成 placement description"""
        from vllm import SamplingParams  # type: ignore

        prompts = []
        for desc in descs:
            prompt = (
                "第一张图是包含所有物体的原始房间图，"
                "第二张图是移除了某个物体后的房间图，"
                "第三张图是被移除的物体的参考图。"
                f"请对比这三张图，用简短的中文描述被移除的物体{desc}"
                "原来放在什么位置，以及周围参照物的关系。"
                "以 '请你将[物体名称]摆放在' 开头。"
            )
            prompts.append(prompt)

        # 构建多模态输入列表
        vllm_inputs = []
        for i in range(len(descs)):
            # 使用 processor 的 apply_chat_template 自动生成正确格式的 prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_images[i]},
                        {"type": "image", "image": plane_images[i]},
                        {"type": "image", "image": object_images[i]},
                        {
                            "type": "text",
                            "text": (
                                "第一张图是包含所有物体的原始房间图，"
                                "第二张图是移除了某个物体后的房间图，"
                                "第三张图是被移除的物体的参考图。"
                                f"请对比这三张图，用简短的中文描述被移除的物体{descs[i]}"
                                "原来放在什么位置，以及周围参照物的关系。"
                                "以 '请你将[物体名称]摆放在' 开头。"
                            ),
                        }
                    ]
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            vllm_input = {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "image": [
                        Image.fromarray(original_images[i]),
                        Image.fromarray(plane_images[i]),
                        Image.fromarray(object_images[i]),
                    ]
                },
            }
            vllm_inputs.append(vllm_input)

        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256,
        )

        outputs = self._vllm_engine.generate(vllm_inputs, sampling_params, use_tqdm=False)
        results = [out.outputs[0].text.strip() for out in outputs]

        return results

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

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            images=[original_pil, plane_pil, obj_pil],
            return_tensors="pt",
        ).to(self._model.device)

        from transformers import GenerationConfig  # type: ignore
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=256,
                    do_sample=False,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        result = self._processor.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()

        logger.info(f"transformers placement description: {result[:80]}...")
        return result

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

    def generate_responses_batch(
        self,
        original_images: List[np.ndarray],
        object_images: List[np.ndarray],
        text_prompts: List[str],
        rotation_6d_list: List[List[float]],
        scale_list: List[float],
    ) -> List[str]:
        """
        批量生成回复（多模态）。

        Args:
            original_images: 原始房间图列表
            plane_images: 平面图列表
            object_images: 物体参考图列表
            text_prompts: 放置指令列表
            rotation_6d_list: 6D 旋转表示列表
            scale_list: 缩放比例列表

        Returns:
            回复列表
        """
        if self.use_vllm:
            return self._vllm_generate_responses_batch(
                original_images, object_images,
                text_prompts, rotation_6d_list, scale_list
            )
        else:
            results = []
            for i in range(len(text_prompts)):
                result = self._transformers_generate_response(
                    original_images[i], object_images[i],
                    text_prompts[i], rotation_6d_list[i], scale_list[i]
                )
                results.append(result)
            return results

    def _vllm_generate_responses_batch(
        self,
        original_images: List[np.ndarray],
        object_images: List[np.ndarray],
        text_prompts: List[str],
        rotation_6d_list: List[List[float]],
        scale_list: List[float],
    ) -> List[str]:
        """vLLM 批量生成回复（多模态）"""
        from vllm import SamplingParams  # type: ignore

        # 构建多模态输入列表
        vllm_inputs = []
        for i in range(len(text_prompts)):
            rot_y_deg = self.extract_rotation_y(rotation_6d_list[i])
            
            # 使用 processor 的 apply_chat_template 生成正确格式的 prompt
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "你是3D物体摆放助手。两张输入图都是俯视图（从上方垂直向下看），"
                                "方向统一按图中视角定义：上方=前，下方=后，左方=左，右方=右。\n\n"
                                "【观察步骤】\n"
                                "1. 看物体参考图（第二张）：判断物体正面当前朝哪个方向（前/后/左/右）\n"
                                "2. 看原始平面图（第一张）：结合指令判断目标摆放方向（前/后/左/右）\n"
                                "3. 旋转只绕垂直轴(Y轴)，正角度=顺时针(向右转)，负角度=逆时针(向左转)\n\n"
                                "【回复格式】\n"
                                "'好的，我会将[物体]摆放在[具体位置]。'\n"
                                "'物体在参考图中朝[前/后/左/右]，在场景中需要朝[前/后/左/右]摆放，且参考大小[偏大/偏小/正常]'\n"
                                "'所以要绕Y轴旋转[角度]°，缩放[比例]倍。'\n"
                                "'综上所述，我会把物体放在<SEG>'\n\n"
                                "【重要】句末必须包含<SEG>，不要添加任何其他内容。"
                                "【方向转换表】物体默认朝前时：\n"
                                "- 目标朝前 → 0°\n"
                                "- 目标朝后 → 180° 或 -180°\n"
                                "- 目标朝右 → +90°（顺时针）\n" 
                                "- 目标朝左 → -90°（逆时针）\n"
                                "【缩放转换表】"
                                " 物体参考大小偏大时，要缩小，缩放比例小于1\n"
                                " 物体参考大小偏小时，要放大，缩放比例大于1\n"
                                " 物体参考大小正常时，缩放比例约等于1\n"
                                "如果物体当前不是朝前，请先计算当前方向与目标方向的差值，再查表。\n"
                                "严禁使用'上/下'描述水平朝向，必须用'前/后/左/右'。"
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": original_images[i]},      # 平面图
                        {"type": "image", "image": object_images[i]},      # 物体参考图
                        {
                            "type": "text",
                            "text": (
                                f"原始平面图<image>，物体参考图<image>\n"
                                f"放置指令：{text_prompts[i]}\n"
                                f"已计算好的精确变换：绕Y轴旋转 {rot_y_deg:.1f}°，缩放 {scale_list[i]:.2f} 倍\n"
                                f"请根据两张俯视图判断物体的当前朝向和目标朝向，"
                                f"并用上述精确参数生成回复。"
                            )
                        }
                    ]
                }
            ]

            text_prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            vllm_input = {
                "prompt": text_prompt,
                "multi_modal_data": {
                    "image": [
                        Image.fromarray(original_images[i]),
                        Image.fromarray(object_images[i]),
                    ]
                },
            }
            vllm_inputs.append(vllm_input)

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.85,
            max_tokens=256,
            stop=["<SEG>"],
        )

        outputs = self._vllm_engine.generate(vllm_inputs, sampling_params, use_tqdm=False)
        results = [out.outputs[0].text.strip() for out in outputs]

        # 确保 <SEG> 结尾
        for i, result in enumerate(results):
            if not result.endswith("<SEG>"):
                results[i] = result.rstrip() + "<SEG>"

        return results

    def _transformers_generate_response(
        self,
        original_image: np.ndarray,
        object_image: np.ndarray,
        text_prompt: str,
        rotation_6d: List[float],
        scale: float,
    ) -> str:
        """transformers 生成回复（多模态）"""
        rot_y_deg = self.extract_rotation_y(rotation_6d)

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "你是3D物体摆放助手。两张输入图都是俯视图（从上方垂直向下看），"
                            "方向统一按图中视角定义：上方=前，下方=后，左方=左，右方=右。\n\n"
                            "【观察步骤】\n"
                            "1. 看物体参考图（第二张）：判断物体正面当前朝哪个方向（前/后/左/右）\n"
                            "2. 看原始平面图（第一张）：结合指令判断目标摆放方向（前/后/左/右）\n"
                            "3. 旋转只绕垂直轴(Y轴)，正角度=顺时针(向右转)，负角度=逆时针(向左转)\n\n"
                            "【回复格式】\n"
                            "'好的，我会将[物体]摆放在[具体位置]。'\n"
                            "'物体在参考图中朝[前/后/左/右]，在场景中需要朝[前/后/左/右]摆放，且参考大小[偏大/偏小/正常]'\n"
                            "'所以要绕Y轴旋转[角度]°，缩放[比例]倍。'\n"
                            "'综上所述，我会把物体放在<SEG>'\n\n"
                            "【重要】句末必须包含<SEG>，不要添加任何其他内容。"
                            "【方向转换表】物体默认朝前时：\n"
                            "- 目标朝前 → 0°\n"
                            "- 目标朝后 → 180° 或 -180°\n"
                            "- 目标朝右 → +90°（顺时针）\n" 
                            "- 目标朝左 → -90°（逆时针）\n"
                            "【缩放转换表】"
                            " 物体参考大小偏大时，要缩小，缩放比例小于1\n"
                            " 物体参考大小偏小时，要放大，缩放比例大于1\n"
                            " 物体参考大小正常时，缩放比例约等于1\n"
                            "如果物体当前不是朝前，请先计算当前方向与目标方向的差值，再查表。\n"
                            "严禁使用'上/下'描述水平朝向，必须用'前/后/左/右'。"
                        )
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": original_image},      # 平面图
                    {"type": "image", "image": object_image},      # 物体参考图
                    {
                        "type": "text",
                        "text": (
                            f"原始平面图<image>，物体参考图<image>\n"
                            f"放置指令：{text_prompt}\n"
                            f"已计算好的精确变换：绕Y轴旋转 {rot_y_deg:.1f}°，缩放 {scale:.2f} 倍\n"
                            f"请根据两张俯视图判断物体的当前朝向和目标朝向，"
                            f"并用上述精确参数生成回复。"
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
            images=[
                Image.fromarray(original_image),
                Image.fromarray(object_image),
            ],
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

        return result
