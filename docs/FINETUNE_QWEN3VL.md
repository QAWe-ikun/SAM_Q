# 微调 Qwen3-VL 输出 [SEG] Token

## 概述

SAM-Q 使用 Qwen3-VL 作为视觉语言编码器，通过 `[SEG]` token 桥接到 SAM3 解码器。
微调的目标是让 Qwen3-VL 理解文本指令 + 多模态图像后，在正确的位置生成 `[SEG]` token。

---

## 架构

```
输入:
  - plane_image (房间俯视图)
  - object_image (物体俯视图)
  - text_prompt ("把椅子放在桌子旁边")

Qwen3-VL (LoRA)
  ↓
  处理 <image> placeholders + 文本
  ↓
  输出 hidden states → 找到 [SEG] token 位置
  ↓
SEG Token Projector
  ↓
  投影到 SAM3 空间 (256-dim)
  ↓
SAM3 Decoder
  ↓
  输出 placement heatmap (64x64)
```

---

## 1. 准备微调数据

### 数据格式

需要构建 **instruction tuning** 格式的数据集，类似 SA2VA：

```json
[
  {
    "id": "sample_001",
    "images": ["plane_images/scene_001.png", "object_images/chair_001.png"],
    "conversation": [
      {
        "from": "human",
        "value": "<image>\n<image>\n把椅子放在桌子旁边"
      },
      {
        "from": "gpt",
        "value": "好的，我会在桌子旁边放置椅子。<SEG>"
      }
    ],
    "mask_path": "masks/scene_001_chair.png",
    "split": "train"
  }
]
```

### 关键点

- `<image>` 是 Qwen3-VL 的图像占位符，**必须**按顺序出现
- `[SEG]` 出现在 GPT 回复的**末尾**，表示生成分割 token
- 每个样本可以有**多个** `[SEG]`（多物体/多位置场景）

---

## 2. 配置 LoRA 微调

### 修改 `configs/config.yaml`

```yaml
model:
  num_seg_tokens: 1  # 1=单SEG, >1=多SEG([SEG0]~[SEG{n-1}])

  qwen:
    model_name: "./models/qwen3_vl"
    hidden_dim: 4096
    freeze: false  # ← 改为 false！
    attn_implementation: "flash_attention_2"

    # LoRA 配置 (新增)
    lora:
      enabled: true
      r: 64
      alpha: 128
      dropout: 0.05
      target_modules:
        - "q_proj"
        - "k_proj"
        - "v_proj"
        - "o_proj"
        - "gate_proj"
        - "up_proj"
        - "down_proj"
      use_qlora: false  # QLoRA 4-bit: 设为 true (显存<12GB时)
      bias: "none"

  sam3:
    freeze_image_encoder: true  # SAM3 图像编码器保持冻结
    freeze_detector: false      # 解码器可训练

training:
  num_epochs: 100
  batch_size: 2        # LoRA 可以小 batch
  lr: 1.0e-4           # LoRA 学习率 (比全参数大)
```

### 显存需求

| 模式 | 显存 (RTX 4090) |
|------|-----------------|
| FP16 + LoRA | ~21 GB |
| QLoRA 4-bit | ~10 GB |

---

## 3. 创建微调数据集

### 使用 `VLADataset` (已有)

```python
# src/data/vla_dataset.py 已经支持多模态输入
from src.data.vla_dataset import VLADataset

dataset = VLADataset(
    data_dir="data/",
    split="train",
    max_seq_length=2048,  # Qwen3-VL 最大上下文
)
```

### 或自定义 Dataset

```python
from torch.utils.data import Dataset
from PIL import Image
import json

class SEGFinetuneDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        with open(f"{data_dir}/annotations.json") as f:
            self.data = json.load(f)
        self.data = [x for x in self.data if x.get("split") == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 加载图像
        plane_img = Image.open(item["plane_image"]).convert("RGB")
        object_img = Image.open(item["object_image"]).convert("RGB")

        # 构建 prompt
        text_prompt = item["text_prompt"]

        return {
            "plane_image": plane_img,
            "object_image": object_img,
            "text_prompt": text_prompt,
            "mask": load_mask(item["mask_path"]),
        }
```

---

## 4. 运行微调

### 命令

```bash
# LoRA 微调 (FP16, ~21GB VRAM)
python main.py train --config configs/config.yaml

# QLoRA 微调 (4-bit, ~10GB VRAM)
# 修改 config.yaml 中 use_qlora: true
python main.py train --config configs/config.yaml
```

### 训练流程

```python
# 伪代码
model = SAMQPlacementModel(
    qwen_model_name="./models/qwen3_vl",
    freeze_qwen=False,  # ← 关键！
)

# 启用 LoRA
model.encoder.enable_lora(
    lora_r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]
)

# 训练
for batch in dataloader:
    output = model(
        plane_image=batch["plane_image"],
        text_prompt=batch["text_prompt"],
        images=[batch["plane_image"], batch["object_image"]],
    )

    loss = criterion(
        output["heatmap"], batch["mask"],
        output.get("rotation_6d"),
        output.get("scale_relative"),
    )
    loss.backward()
    optimizer.step()
```

---

## 5. 保存和加载

### 保存 LoRA 权重

```python
# 训练结束后自动保存
outputs/checkpoint_best.pt  # 包含 LoRA adapter 权重
```

### 加载微调后的模型

```python
model = SAMQPlacementModel(
    qwen_model_name="./models/qwen3_vl",
    checkpoint_path="outputs/checkpoint_best.pt",
)

# 或者只加载 LoRA adapter
from peft import PeftModel

base_model = AutoModel.from_pretrained("./models/qwen3_vl")
lora_model = PeftModel.from_pretrained(
    base_model,
    "outputs/lora_adapter/"
)
```

### 合并 LoRA 权重 (推理时)

```python
model.encoder.merge_lora()  # 合并到基础模型，加速推理
```

---

## 6. 验证 [SEG] 输出

### 测试生成

```python
output = model.predict(
    plane_image=room_img,
    text_prompt="把椅子放在桌子旁边",
    images=[room_img, chair_img],
)

# 检查内部 [SEG] token 生成
seg_hidden_states = output["seg_hidden_states"]
print(f"SEG token hidden state shape: {seg_hidden_states.shape}")
# Expected: [B, num_seg_tokens, 4096]
```

### 可视化注意力

```python
# 查看 Qwen3-VL 对 <image> 的注意力
attn_weights = output["cross_attention_weights"]
# 应该看到模型关注 text_prompt 提到的物体位置
```

---

## 常见问题

### Q: 需要多少数据？

**A:** LoRA 微调通常 **100-1000** 个高质量样本即可。关键是要有：
- 多样化的房间布局
- 多样化的物体类型
- 清晰的位置描述文本

### Q: 训练多长时间？

**A:** 取决于显存和数据量：
- RTX 4090, 500 samples, 3 epochs: **~2-4 小时**
- QLoRA 4-bit 可以更快 (~1.5x)

### Q: 如何判断 [SEG] token 是否正确学习？

**A:** 观察训练 loss：
- `dice_loss` 下降 → heatmap 质量提升
- 可视化 heatmap 应该集中在合理的位置

### Q: 可以只微调 [SEG] token 相关层吗？

**A:** 可以，但当前实现是标准的 LoRA（7 个模块）。如果要只微调特定层，修改 `target_modules`。

---

## 参考

- **SA2VA**: 使用 [SEG] token 桥接 VLM 和分割模型
- **Qwen3-VL**: 多模态大语言模型，支持 `<image>` placeholders
- **LoRA**: Low-Rank Adaptation, 高效微调技术
