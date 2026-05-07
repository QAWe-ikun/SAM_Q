"""
SEG Feature Extractor
=====================

Extracts <SEG> token hidden states from Qwen3-VL after Stage 1 training.
"""

import gc
import torch # type: ignore
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader  # type: ignore


class SegFeatureExtractor:
    """
    Extracts <SEG> hidden states for Stage 2 training.
    Uses batching and memory cleanup to prevent OOM.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
        batch_size: int = 4,
    ):
        self.model = model
        self.config = config
        self.batch_size = batch_size

    def extract(self, dataloader: DataLoader, output_dir: Path) -> None:
        """
        Extract <SEG> features and save to disk.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.qwen_encoder.load_model(use_cache=True)

        print(f"\n{'=' * 60}")
        print(f"提取 <SEG> features → {output_dir}")
        print(f"{'=' * 60}")

        dataset = dataloader.dataset
        samples = dataset.samples

        system_prompt = getattr(self.model.qwen_encoder, 'system_prompt', '')
        tokenizer = self.model.qwen_encoder.processor.tokenizer
        device = self.model.qwen_encoder.device
        seg_id = self.model.qwen_encoder.seg_token_id
        count = 0

        for start_idx in tqdm(range(0, len(samples), self.batch_size), desc="提取 <SEG>"):
            end_idx = min(start_idx + self.batch_size, len(samples))
            gt_items = []      # (sample_id, out_path, text, img_list)
            gen_items = []     # (sample_id, out_path, text_prompt, images)

            for idx in range(start_idx, end_idx):
                sample = dataset[idx]
                ann = samples[idx]
                sample_id = ann.get("sample_id") or ann.get("id", f"sample_{idx:06d}")

                split = ann.get("split", dataset.split)
                scene_dir = ann.get("scene_dir", "")
                scene_id = Path(scene_dir).name if scene_dir else "unknown"
                out_dir = output_dir / split / scene_id
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{sample_id}.pt"

                if out_path.exists():
                    count += 1
                    continue

                text_prompt = sample["text_prompt"]
                response = sample.get("response", None)
                images = sample.get("images", [])

                if response is not None:
                    messages, img_list = self.model.qwen_encoder._build_message(
                        text_prompt=text_prompt,
                        images=images,
                    )
                    if system_prompt:
                        messages = [
                            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                        ] + messages
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
                    text = self.model.qwen_encoder.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    gt_items.append((sample_id, out_path, text, img_list))
                else:
                    gen_items.append((sample_id, out_path, text_prompt, images))

            # 批量处理有 response 的样本
            if gt_items:
                batch_texts = [t for _, _, t, _ in gt_items]
                batch_images = [il if il else None for _, _, _, il in gt_items]
                has_images = any(imgs is not None for imgs in batch_images)
                inputs = self.model.qwen_encoder.processor(
                    text=batch_texts,
                    images=batch_images if has_images else None,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    outputs = self.model.qwen_encoder.model(
                        **inputs,
                        output_hidden_states=True,
                    )
                    hidden_states = outputs.hidden_states

                for i, (sample_id, out_path, _, _) in enumerate(gt_items):
                    if out_path.exists():
                        count += 1
                        continue

                    mask = inputs["input_ids"][i] == seg_id
                    if mask.any():
                        seg_pos = mask.nonzero(as_tuple=False)[:, 0]
                        seg_features = hidden_states[-1][i][seg_pos].cpu().float()
                    else:
                        seg_features = None

                    torch.save({"seg_hidden": seg_features, "sample_id": sample_id}, out_path)
                    count += 1

                del outputs, hidden_states, inputs

            # 逐个处理没有 response 的样本（先生成 response，再提取所有层）
            for sample_id, out_path, text_prompt, images in gen_items:
                if out_path.exists():
                    count += 1
                    continue

                # 先 model.generate 得到完整的 response 文本
                messages, img_list = self.model.qwen_encoder._build_message(
                    text_prompt=text_prompt,
                    images=images,
                )
                if system_prompt:
                    messages = [
                        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
                    ] + messages

                gen_text = self.model.qwen_encoder.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                gen_inputs = self.model.qwen_encoder.processor(
                    text=[gen_text],
                    images=img_list if img_list else None,
                    return_tensors="pt",
                ).to(device)

                generated_ids = self.model.qwen_encoder.model.generate(
                    **gen_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    eos_token_id=seg_id,
                )
                generated_text = tokenizer.decode(generated_ids[0, gen_inputs["input_ids"].shape[1]:], skip_special_tokens=False)
                del gen_inputs

                # 构建完整输入（prompt + 生成的 response）
                messages.append({"role": "assistant", "content": [{"type": "text", "text": generated_text}]})
                full_text = self.model.qwen_encoder.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                full_inputs = self.model.qwen_encoder.processor(
                    text=[full_text],
                    images=img_list if img_list else None,
                    return_tensors="pt",
                ).to(device)
                del full_text, messages, generated_ids

                with torch.no_grad():
                    gen_outputs = self.model.qwen_encoder.model(
                        **full_inputs,
                        output_hidden_states=True,
                    )
                    gen_hidden = gen_outputs.hidden_states
                    gen_mask = full_inputs["input_ids"][0] == seg_id
                    if gen_mask.any():
                        seg_pos = gen_mask.nonzero(as_tuple=False)[:, 0]
                        seg_features = gen_hidden[-1][0][seg_pos].cpu().float()
                    else:
                        seg_features = None

                torch.save({"seg_hidden": seg_features, "sample_id": sample_id}, out_path)
                count += 1

                del gen_outputs, gen_hidden, full_inputs
                torch.cuda.empty_cache()
                gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

        print(f"完成: {count} 个特征已保存到 {output_dir}")
