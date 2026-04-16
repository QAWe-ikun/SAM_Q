# 训练指南：两阶段训练 Qwen3-VL + SAM3 Decoder

## 概述

SAM-Q 使用 Qwen3-VL 作为视觉语言编码器，通过 `<SEG>` token 桥接到 SAM3 解码器。

**训练分为两个阶段**：

1. **阶段 1**: 微调 Qwen3-VL 输出 `<SEG>` token（理解文本指令 + 图像 → 生成分割 token）
2. **阶段 2**: 训练 Adapter + SAM3 Decoder（将 `<SEG>` hidden state 映射到 heatmap + rotation + scale）

为什么分两阶段？
- Qwen3-VL 初始不知道 `<SEG>` token 的含义
- 如果直接端到端训练，SAM3 Decoder 接收到无意义的 hidden state，梯度会爆炸
- 先让 Qwen3-VL 学会在合适位置生成 `<SEG>`，再训练 Decoder 学习物理空间映射
- Stage 1 训练结束后**自动提取** `<SEG>` hidden states 到 `data/seg_features/`，Stage 2 直接读取，**无需加载 Qwen3-VL**（节省 ~16GB 显存）

---

## 数据格式

### annotations.json

所有样本在一个 JSON 数组中，通过 `split` 字段区分训练/验证/测试。

```json
[
  {
    "scene_id": "scene_001",
    "split": "train",
    "plane_image_path": "plane_images/scene_001.png",
    "images_path": [
      "plane_images/scene_001.png",
      "object_images/chair_001.png"
    ],
    "mask_path": "masks/scene_001_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放在桌子旁边",
    "response": "好的，我会在桌子旁边放置椅子。<SEG>",
    "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "scale": 1.0
  }
]
```

### 字段说明

| 字段 | 类型 | Stage 1 | Stage 2 | 说明 |
|------|------|---------|---------|------|
| `plane_image_path` | str | 必须 | 必须 | SAM3 输入（房间/平面俯视图） |
| `images_path` | list[str] | 必须 | 必须 | Qwen3-VL 输入的图片列表，**顺序必须与 `<image>` 占位符一致** |
| `mask_path` | str | 必须 | 必须 | 二值 mask 图，白色 = 放置区域 |
| `text_prompt` | str | 必须 | 必须 | 放置指令，支持 `<image>` 占位符 |
| `response` | str | **必须** | 不需要 | Qwen3-VL 的目标回复，末尾包含 `<SEG>` |
| `rotation_6d` | list[6] | 不需要 | **必须** | 6D 旋转表示 |
| `scale` | float | 不需要 | **必须** | 缩放比例，1.0 = 原始大小 |
| `split` | str | 可选 | 可选 | `"train"` / `"val"` / `"test"` |
| `scene_id` | str | 可选 | 可选 | 场景标识，用于 seg_features 文件名 |

### images_path 设计原则

```json
{
  "plane_image_path": "plane_images/scene_001.png",       // SAM3 专用
  "images_path": [                                         // Qwen3-VL 多模态输入
    "plane_images/scene_001.png",                           // 第1张 → 第1个 <image>
    "object_images/chair_001.png"                           // 第2张 → 第2个 <image>
  ]
}
```

- `plane_image_path` 单独列出，因为 SAM3 只处理房间/平面俯视图
- `images_path` 包含所有要传给 Qwen3-VL 的图片，**顺序必须与 `text_prompt` 中 `<image>` 的出现顺序一致**
- 通常 `images_path[0] == plane_image_path`

---

## 架构

```
阶段 1: Qwen3-VL 微调
─────────────────────────────────
输入:
  - images_path 中的所有图片（按 <image> 顺序）
  - text_prompt（含 <image> 占位符）
  - response（含 <SEG> 的 GPT 回复）

Qwen3-VL (LoRA)
  ↓
  处理 <image> placeholders + 文本
  ↓
  输出 token sequence → 包含 <SEG>
  ↓
Loss: 语言模型 next-token prediction

自动后处理: 提取 <SEG> hidden state → data/seg_features/{scene_id}.pt

─────────────────────────────────

阶段 2: Adapter + SAM3 Decoder 训练（不加载 Qwen3-VL）
─────────────────────────────────
输入:
  - plane_image_path → SAM3（房间俯视图）
  - seg_features（预提取的 <SEG> hidden state）

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
Loss: Dice + BCE (heatmap) + rotation L1 + scale L1
```

---

## 阶段 1: 微调 Qwen3-VL 输出 <SEG> Token

### 目标

让 Qwen3-VL 理解：
- 输入文本指令 + 多模态图像
- 在回复中正确位置生成 `<SEG>` token

### 1.1 准备数据

```json
[
  {
    "scene_id": "scene_001",
    "split": "train",
    "plane_image_path": "plane_images/scene_001.png",
    "images_path": ["plane_images/scene_001.png", "object_images/chair_001.png"],
    "mask_path": "masks/scene_001_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放在桌子旁边",
    "response": "好的，我会在桌子旁边放置椅子。<SEG>",
    "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "scale": 1.0
  }
]
```

**关键点**：
- `images_path` 顺序必须与 `text_prompt` 中 `<image>` 的出现顺序一致
- `response` 必须包含自然语言描述 + `<SEG>` 在末尾
- Stage 1 训练时必须提供 `response` 字段

### 1.2 配置

使用 `configs/stage1_qwen_lora.yaml`：

```yaml
model:
  num_seg_tokens: 1

  qwen:
    model_name: "./models/qwen3_vl"
    hidden_dim: 4096
    freeze: false  # ← 不冻结，启用 LoRA
    attn_implementation: "flash_attention_2"

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
      use_qlora: false  # 显存<12GB时设为 true
      bias: "none"

  sam3:
    checkpoint_path: null  # ← Stage 1 不加载 SAM3

  adapter:
    freeze: true  # ← 冻结

loss:
  type: "lm"  # 语言模型损失

optimizer:
  lr: 1.0e-4

training:
  num_epochs: 3  # 通常 2-5 个 epoch 足够
  batch_size: 1  # LoRA 训练建议 batch_size=1
  gradient_accumulation_steps: 8  # 模拟更大的 batch size
```

### 1.3 运行

```bash
python main.py train --config configs/stage1_qwen_lora.yaml
```

**输出**：
- `outputs/stage1_lora/checkpoint_best.pt` — LoRA 权重
- `data/seg_features/*.pt` — 自动提取的 `<SEG>` hidden states

### 1.4 验证 <SEG> 输出

训练结束后会自动运行一次验证，打印生成的文本对比：

```
============================================================
Running Stage 1 Validation (Check <SEG> generation)...
============================================================
Generated: 好的，我会在桌子旁边放置椅子。<SEG>
Expected:  好的，我会在桌子旁边放置椅子。<SEG>
============================================================
```

也可以手动测试：

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

训练 `<SEG>` hidden state → 物理空间映射：
- **CrossModalAdapter**: <SEG> hidden state → SAM3 prompt embeddings
- **SAM3 Transformer Decoder**: 生成 heatmap
- **SEGActionHead**: 预测 rotation_6d + scale_relative

### 2.1 配置

使用 `configs/stage2_decoder.yaml`：

```yaml
model:
  num_seg_tokens: 1

  qwen:
    model_name: "./models/qwen3_vl"  # ← 有 seg_features 时不加载
    lora_path: "outputs/stage1_lora/"  # ← Stage 1 LoRA（可选）
    hidden_dim: 4096
    freeze: true  # ← 冻结 Qwen3-VL
    attn_implementation: "flash_attention_2"

  sam3:
    checkpoint_path: "./models/sam3/sam3.pt"  # ← 指向 .pt 文件
    input_dim: 256
    freeze_image_encoder: true  # SAM3 image encoder frozen
    freeze_detector: false      # ← SAM3 decoder trainable

  adapter:
    freeze: false  # ← Trainable
    hidden_dim: 512
    num_queries: 64
    dropout: 0.1

  seg_token:
    projector_hidden_dim: 1024  # 4096 → 1024 → 256
    dropout: 0.1

  action_head:
    heatmap_size: 64

loss:
  type: "placement"
  dice_weight: 1.0
  bce_weight: 1.0
  rotation_weight: 0.5
  scale_weight: 0.3

optimizer:
  type: "AdamW"
  lr: 1.0e-3  # Full parameter learning rate (larger than LoRA)
  weight_decay: 1.0e-4
  betas: [0.9, 0.999]

scheduler:
  type: "CosineAnnealingLR"
  T_max: 50
  eta_min: 1.0e-6
  warmup_epochs: 5

training:
  num_epochs: 50
  device: "cuda"
  save_dir: "outputs/stage2_full/"
  save_best: true
  save_epoch: false  # Set true to save at save_interval epochs
  save_interval: 10  # Save every N epochs (when save_epoch=true)
  log_interval: 10
  val_interval: 1
  early_stopping: true
  patience: 20  # 连续 20 个 epoch val_loss 不下降则停止

data:
  seg_feature_dir: "data/seg_features/"  # ← 读取 Stage 1 提取的特征
  batch_size: 4
  num_workers: 4
  pin_memory: true
```

### 2.2 运行

```bash
python main.py train --config configs/stage2_decoder.yaml
```

**训练日志示例**：

```
============================================================
[Stage 2] Trainable Parameters Check:
============================================================
  [Train] adapter.input_proj.weight: 2,097,152
  [Train] seg_action_head.heatmap_head.0.weight: 2,097,152
  [Train] sam3_loader.model.transformer.decoder.layers.0.cross_attn...
[Stage 2] Total Trainable Params: ~65,000,000

Epoch   1 | train_loss: 1.4847 | bce: 0.6931 | dice: 0.7916 | rot: 0.1234 | scl: 0.0567 | val_loss: 3.3697 | iou: 0.0097 | lr: 0.000200
Epoch   2 | train_loss: 1.2345 | bce: 0.5678 | dice: 0.6667 | rot: 0.0987 | scl: 0.0432 | val_loss: 2.1234 | iou: 0.1234 | lr: 0.000200
...
```

**日志说明**：

| 指标 | 说明 | 预期趋势 |
|------|------|----------|
| `train_loss` | 总训练损失（加权和） | 逐渐下降 |
| `bce` | 二分类交叉熵（像素级分类） | 逐渐下降 |
| `dice` | Dice Loss（区域重叠度） | 逐渐下降 |
| `rot` | Rotation Loss（6D 旋转预测） | 有 GT 标注时下降 |
| `scl` | Scale Loss（相对缩放预测） | 有 GT 标注时下降 |
| `val_loss` | 验证损失 | 先降后平（过拟合信号） |
| `iou` | Intersection over Union | 逐渐上升 |

**输出**：
- `outputs/stage2_full/checkpoint_best.pt` — Adapter + SAM3 Decoder 权重
- `outputs/stage2_full/checkpoint_final.pt` — 最后一个 epoch 的权重

---

## 推理

### 命令行

```bash
python main.py predict \
  --checkpoint outputs/stage2_full/checkpoint_best.pt \
  --config configs/config.yaml \
  --plane_image data/plane_images/scene_001.png \
  --object_image data/object_images/chair_001.png \
  --prompt "把椅子放在桌子旁边" \
  --output outputs/
```

### 参数说明

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--checkpoint` | **必需**：训练好的模型权重文件 | `outputs/stage2_full/checkpoint_best.pt` |
| `--config` | 推理配置（可选） | `configs/config.yaml` |
| `--plane_image` | **必需**：房间俯视图 | `data/plane_images/scene_001.png` |
| `--object_image` | **必需**：物体俯视图 | `data/object_images/chair_001.png` |
| `--prompt` | 文本指令（可选） | `"把椅子放在合适的位置"` |
| `--threshold` | 置信度阈值（可选，默认 0.5） | `0.5` |
| `--output` | 输出目录（可选） | `outputs/` |

### 预期输出

运行后会在 `outputs/` 目录下生成：

```
outputs/
├── prediction.png      # 可视化结果（heatmap 叠加在原图上）
└── prediction.json     # 结构化预测结果（heatmap shape、rotation、scale 等）
```

### Python API

```python
from src.inference.predictor import PlacementPredictor
from PIL import Image

# 加载模型
predictor = PlacementPredictor(
    checkpoint_path="outputs/stage2_full/checkpoint_best.pt",
    device="cuda",
    threshold=0.5,
)

# 预测
results = predictor.predict(
    plane_image=Image.open("data/plane_images/scene_001.png"),
    object_image=Image.open("data/object_images/chair_001.png"),
    text_prompt="把椅子放在桌子旁边",
)

# 结果
print(f"Heatmap shape: {results['heatmap'].shape}")
print(f"Rotation: {results['rotation_deg']}°")
print(f"Scale: {results['scale_relative']}x")

# 可视化
overlay = predictor.visualize(
    plane_image=Image.open("data/plane_images/scene_001.png"),
    results=results,
)
overlay.save("prediction_overlay.png")
```

---

## 常见问题

### Q: 为什么不端到端训练？

**A:** 端到端训练会导致：
1. Qwen3-VL 初始不知道 `<SEG>` 含义，hidden state 无意义
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
1. LoRA loss 下降 → Qwen3-VL 学会预测 `<SEG>`
2. 手动测试生成，检查是否输出 `<SEG>` token
3. 如果输出中没有 `<SEG>`，可能是数据格式不对或 epoch 不够

### Q: 阶段 2 的 `Recall = 1.0` 但 `Precision ≈ 0` 是怎么回事？

**A:** 这是训练初期的正常现象。说明模型倾向于把所有区域都预测为掩码（"全白"预测）。随着训练进行，Precision 会上升，Recall 会趋于合理。只要 Loss 在下降就无需担心。

### Q: 如何只训练 SAM3 Decoder 而不训练其他部分？

**A:** 在配置中设置：
```yaml
sam3:
  freeze_image_encoder: true   # 冻结图像编码器
  freeze_detector: false       # 解冻 decoder

adapter:
  freeze: false  # Adapter 也训练

qwen:
  freeze: true  # Qwen 冻结
```

### Q: `seg_features` 必须存在吗？

**A:** **不是必须的**，但强烈建议使用。
- 如果有 `data/seg_feature_dir` 配置，Stage 2 会**跳过 Qwen3-VL 加载**（节省 ~16GB 显存）
- 如果没有，Stage 2 会在线推理 Qwen3-VL 生成 `<SEG>`，速度更慢且需要更多显存

---

## 完整训练流程

```bash
# ─────────────────────────────────────────────
# 阶段 1: Qwen3-VL LoRA 微调
# 训练结束后自动提取 <SEG> features 到 data/seg_features/
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
# 推理
# ─────────────────────────────────────────────
python main.py predict \
  --checkpoint outputs/stage2_full/checkpoint_best.pt \
  --config configs/config.yaml \
  --plane_image data/plane_images/scene_001.png \
  --object_image data/object_images/chair_001.png \
  --prompt "把椅子放在桌子旁边" \
  --output outputs/
```

---

## 参考

- **SA2VA**: 使用 <SEG> token 桥接 VLM 和分割模型
- **Qwen3-VL**: 多模态大语言模型，支持 `<image>` placeholders
- **LoRA**: Low-Rank Adaptation, 高效微调技术
- **SAM3**: Segment Anything Model 3, 通用分割模型
