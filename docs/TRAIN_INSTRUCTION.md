# 训练指南

## 概述

SAM-Q 使用 Qwen3-VL 作为视觉语言编码器，通过 `[SEG]` token 桥接到 SAM3 解码器。

**训练分为两个阶段**：

1. **阶段 1**: 微调 Qwen3-VL 输出 `[SEG]` token（理解文本指令 + 图像 → 生成分割 token）
2. **阶段 2**: 训练 Adapter + SAM3 Decoder（将 `[SEG]` hidden state 映射到 heatmap + rotation + scale）

为什么分两阶段？
- Qwen3-VL 初始不知道 `[SEG]` token 的含义
- 如果直接端到端训练，SAM3 Decoder 接收到无意义的 hidden state，梯度会爆炸
- 先让 Qwen3-VL 学会在合适位置生成 `[SEG]`，再训练 Decoder 学习物理空间映射

---

## 架构

```
annotations.json 格式:
  plane_image_path: "plane_images/scene_001.png"          # SAM3 输入
  images_path: ["plane_images/scene_001.png",              # Qwen3-VL 输入（按顺序匹配 <image>）
                "object_images/chair_001.png"]
  text_prompt: "<image>\n<image>\n把椅子放在桌子旁边"


阶段 1: Qwen3-VL 微调
─────────────────────────────────
输入:
  - images_path 中的所有图片（按 <image> 顺序）
  - text_prompt（含 <image> 占位符）
  - response（含 [SEG] 的 GPT 回复）

Qwen3-VL (LoRA)
  ↓
  处理 <image> placeholders + 文本
  ↓
  输出 token sequence → 包含 [SEG]
  ↓
Loss: 语言模型 next-token prediction

自动后处理: 提取 [SEG] hidden state → data/seg_features/{scene_id}.pt

─────────────────────────────────

阶段 2: Adapter + SAM3 Decoder 训练
─────────────────────────────────
输入:
  - plane_image_path → SAM3（房间俯视图）
  - seg_features（预提取的 [SEG] hidden state，不加载 Qwen3-VL）

SAM3 Vision Encoder (冻结)
  ↓
  plane_image → 256-dim features
  ↓
SEG Token Projector (可训练) ← seg_hidden [4096-dim]
  ↓
  投影到 SAM3 空间 (256-dim)
  ↓
SAM3 Decoder (可训练)
  ↓
  输出 placement heatmap (64x64)
  + rotation_6d + scale_relative
  ↓
Loss: Dice + BCE (heatmap) + rotation L2 + scale L2
```

---

## 阶段 1: 微调 Qwen3-VL 输出 [SEG] Token

### 目标

让 Qwen3-VL 理解：
- 输入文本指令 + 多模态图像
- 在回复中正确位置生成 `[SEG]` token

### 1.1 准备指令微调数据

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
    "split": "train"
  }
]
```

### 关键点

- `<image>` 是 Qwen3-VL 的图像占位符，**必须**按顺序出现
- `[SEG]` 出现在 GPT 回复的**末尾**，表示生成分割 token
- 每个样本可以有**多个** `[SEG]`（多物体/多位置场景）
- GPT 的回复应该**自然语言描述 + `<SEG>`**，不是只输出 token

### 1.2 配置 LoRA

修改 `configs/config.yaml`：

```yaml
# 阶段 1 配置：只训练 Qwen3-VL
model:
  num_seg_tokens: 1  # 1=单SEG, >1=多SEG([SEG0]~[SEG{n-1}])

  qwen:
    model_name: "./models/qwen3_vl"
    hidden_dim: 4096
    freeze: false  # ← 不冻结，启用 LoRA
    attn_implementation: "flash_attention_2"

    # LoRA 配置
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
      use_qlora: false  # QLoRA 4-bit: 显存<12GB时设为 true
      bias: "none"

  # 阶段 1: SAM3 不需要训练
  sam3:
    freeze_image_encoder: true
    freeze_detector: true  # ← 冻结

  # 阶段 1: Adapter 不需要训练
  adapter:
    freeze: true  # ← 冻结

loss:
  type: "lm"  # 语言模型损失
  label_smoothing: 0.0

optimizer:
  lr: 1.0e-4  # LoRA 学习率

training:
  num_epochs: 3  # 通常 2-5 个 epoch 足够
  batch_size: 2
```

### 显存需求

| 模式 | 显存 (RTX 4090) |
|------|-----------------|
| FP16 + LoRA | ~21 GB |
| QLoRA 4-bit | ~10 GB |

### 1.3 运行阶段 1

```bash
python main.py train --config configs/stage1_qwen_lora.yaml
```

### 1.4 验证 [SEG] 输出

测试 Qwen3-VL 是否正确输出 `[SEG]`：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型 + LoRA
base_model = AutoModelForCausalLM.from_pretrained("./models/qwen3_vl")
lora_model = PeftModel.from_pretrained(base_model, "outputs/stage1_lora/")

# 测试
prompt = "<image>\n<image>\n把椅子放在桌子旁边"
inputs = processor(prompt, images=[plane_img, object_img], return_tensors="pt")
outputs = lora_model.generate(**inputs)
print(processor.decode(outputs[0]))
# 应该输出: "好的，我会在桌子旁边放置椅子。<SEG>"
```

---

## 阶段 2: 训练 Adapter + SAM3 Decoder

### 目标

训练 `[SEG]` hidden state → 物理空间映射：
- **SEG Token Projector**: 4096-dim → 256-dim (SAM3 空间)
- **SAM3 Decoder**: 生成 heatmap + rotation + scale

### 2.1 配置

修改 `configs/config.yaml`：

```yaml
# 阶段 2 配置：训练 Adapter + SAM3 Decoder
model:
  num_seg_tokens: 1

  qwen:
    model_name: "./models/qwen3_vl"
    hidden_dim: 4096
    freeze: true  # ← 冻结 Qwen3-VL（使用阶段 1 的 LoRA）
    lora_checkpoint: "outputs/stage1_lora/"  # ← 加载阶段 1 的 LoRA
    attn_implementation: "flash_attention_2"

  sam3:
    freeze_image_encoder: true  # SAM3 图像编码器保持冻结
    freeze_detector: false      # ← 解码器可训练

  adapter:
    freeze: false  # ← 可训练
    hidden_dim: 512
    num_queries: 64
    dropout: 0.1

  seg_token:
    projector_hidden_dim: 1024  # 4096 → 1024 → 256

loss:
  type: "placement"
  dice_weight: 1.0
  bce_weight: 1.0
  rotation_weight: 0.5
  scale_weight: 0.3

optimizer:
  lr: 1.0e-3  # 全参数学习率（比 LoRA 大）

training:
  num_epochs: 50
  batch_size: 4
```

### 2.2 运行阶段 2

```bash
python main.py train --config configs/stage2_decoder.yaml
```

### 2.3 训练流程

```python
# 阶段 2 伪代码
model = SAMQPlacementModel(
    qwen_model_name="./models/qwen3_vl",
    qwen_lora_checkpoint="outputs/stage1_lora/",  # 加载阶段 1
    freeze_qwen=True,    # Qwen3-VL 冻结
    freeze_sam3_encoder=True,
    freeze_sam3_decoder=False,  # SAM3 decoder 可训练
)

for batch in dataloader:
    output = model(
        plane_image=batch["plane_image"],
        text_prompt=batch["text_prompt"],
        images=[batch["plane_image"], batch["object_image"]],
    )

    # output["seg_hidden_states"]: [B, num_seg_tokens, 4096]
    # output["heatmap"]: [B, num_candidates, H, W]
    # output["rotation_6d"]: [B, 6]
    # output["scale_relative"]: [B]

    loss = criterion(
        output["heatmap"], batch["mask"],
        output.get("rotation_6d"),
        output.get("scale_relative"),
    )
    loss.backward()
    optimizer.step()
```

---

## 常见问题

### Q: 为什么不端到端训练？

**A:** 端到端训练会导致：
1. Qwen3-VL 初始不知道 `[SEG]` 含义，hidden state 无意义
2. SAM3 Decoder 收到随机 hidden state，梯度爆炸
3. 两阶段训练更稳定，且可以复用阶段 1 的 Qwen3-VL LoRA

### Q: 阶段 1 需要多少数据？

**A:** LoRA 微调通常 **100-1000** 个高质量样本即可。关键是要有：
- 多样化的房间布局
- 多样化的物体类型
- 清晰的位置描述文本

### Q: 阶段 1 训练多长时间？

**A:** 取决于显存和数据量：
- RTX 4090, 500 samples, 3 epochs: **~1-2 小时**
- QLoRA 4-bit 可以更快 (~1.5x)

### Q: 阶段 2 需要多少数据？

**A:** 阶段 2 是标准的监督学习，需要：
- **500-5000** 个样本
- 每个样本必须有 ground truth placement mask

### Q: 阶段 2 训练多长时间？

**A:**
- RTX 4090, 1000 samples, 50 epochs: **~4-8 小时**
- SAM3 Decoder 参数量大，比阶段 1 慢

### Q: 如何判断阶段 1 成功？

**A:** 观察：
1. LoRA loss 下降 → Qwen3-VL 学会预测 `[SEG]`
2. 手动测试生成，检查是否输出 `<SEG>` token
3. 如果输出中没有 `<SEG>`，可能是数据格式不对或 epoch 不够

### Q: 可以只微调 [SEG] token 相关层吗？

**A:** 可以，但当前实现是标准的 LoRA（7 个模块）。如果要只微调特定层，修改 `target_modules`。

---

## 完整训练流程示例

```bash
# ─────────────────────────────────────────────
# 阶段 1: Qwen3-VL LoRA 微调
# 训练结束后自动提取 [SEG] features 到 data/seg_features/
# ─────────────────────────────────────────────
python main.py train --config configs/stage1_qwen_lora.yaml
# 输出: outputs/stage1_lora/checkpoint_best.pt
# 自动生成: data/seg_features/*.pt

# ─────────────────────────────────────────────
# 阶段 2: Adapter + SAM3 Decoder 训练
# 直接读取 seg_features，不加载 Qwen3-VL（省 ~16GB 显存）
# ─────────────────────────────────────────────
python main.py train --config configs/stage2_decoder.yaml
# 输出: outputs/stage2_full/checkpoint_best.pt

# ─────────────────────────────────────────────
# 推理: 合并 LoRA + 使用阶段 2 权重
# ─────────────────────────────────────────────
python main.py predict \
  --checkpoint outputs/stage2_full/checkpoint_best.pt \
  --plane_image examples/room.png \
  --object_image examples/chair.png \
  --prompt "把椅子放在桌子旁边" \
  --output results/
```

---

## 参考

- **SA2VA**: 使用 [SEG] token 桥接 VLM 和分割模型
- **Qwen3-VL**: 多模态大语言模型，支持 `<image>` placeholders
- **LoRA**: Low-Rank Adaptation, 高效微调技术
