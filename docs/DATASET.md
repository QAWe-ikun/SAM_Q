# 训练集说明

Stage 1 和 Stage 2 **共用同一个数据集**，统一放在 `data/` 下。
Stage 1 训练结束后自动提取 `<SEG>` hidden states 到 `data/seg_features/`，Stage 2 直接读取，无需加载 Qwen3-VL。

## 目录结构

```
data/
├── annotations.json              # 统一标注文件
├── plane_images/                 # 房间/平面俯视图（SAM3 输入）
│   ├── scene_001.png
│   └── ...
├── object_images/                # 物体俯视图
│   ├── chair_001.png
│   └── ...
├── masks/                        # GT 放置 mask（二值图）
│   ├── scene_001_mask.png
│   └── ...
└── seg_features/                 # <SEG> hidden states（Stage 1 训练后自动生成）
    ├── scene_001.pt
    └── ...
```

---

## annotations.json 格式

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

## 字段说明

| 字段 | 类型 | Stage 1 | Stage 2 | 说明 |
|------|------|---------|---------|------|
| `plane_image_path` | str | 必须 | 必须 | SAM3 输入（房间/平面俯视图） |
| `images_path` | list[str] | 必须 | 必须 | Qwen3-VL 输入的图片列表，**顺序必须与 `<image>` 占位符一致**，第一项通常为 `plane_image_path` |
| `mask_path` | str | 必须 | 必须 | 二值 mask 图，白色 = 放置区域 |
| `text_prompt` | str | 必须 | 必须 | 放置指令，支持 `<image>` 占位符 |
| `response` | str | **必须** | 不需要 | Qwen3-VL 的目标回复，末尾包含 `<SEG>` |
| `rotation_6d` | list[6] | 不需要 | **必须** | 6D 旋转表示（旋转矩阵前两列） |
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

## response 字段规范（Stage 1）

```
# 正确
"好的，我会在桌子旁边放置椅子。<SEG>"
"I will place the chair next to the table. <SEG>"

# 错误
"<SEG>"                              # 没有自然语言描述
"好的，我会放置椅子。<SEG> 完成。"   # <SEG> 不在末尾
```

缺失 `response` 时自动使用默认值：`"好的，我将为您放置物体。<SEG>"`

---

## rotation_6d 说明（Stage 2）

6D rotation 是旋转矩阵 R 的前两列向量拼接：

```
旋转矩阵 R:                    rotation_6d:
| r11  r12  r13 |
| r21  r22  r23 |     →     [r11, r21, r31, r12, r22, r32]
| r31  r32  r33 |
```

常见值：
- **无旋转**: `[1, 0, 0, 0, 1, 0]`
- **绕 Z 轴旋转 90°**: `[0, 1, 0, -1, 0, 0]`
- **绕 Z 轴旋转 45°**: `[0.707, 0.707, 0, -0.707, 0.707, 0]`

---

## seg_features 目录

Stage 1 训练结束后**自动生成**，无需手动创建。

每个 `.pt` 文件包含：
```python
{
    "seg_hidden": torch.Tensor,  # [4096]  Qwen3-VL 在 <SEG> 位置的 hidden state
    "sample_id": str,            # 对应 annotations.json 中的 scene_id
}
```

文件名格式：`{scene_id}.pt`（与 annotations.json 的 `scene_id` 对应）

---

## 图像要求

| 类型 | 格式 | 推荐分辨率 | 说明 |
|------|------|-----------|------|
| plane_image | PNG/JPG, RGB | 1024×1024 | 房间或桌面的俯视图 |
| object_image | PNG/JPG, RGB | 1024×1024 | 物体的俯视图，背景尽量干净 |
| mask | PNG, 灰度 | 与 plane_image 一致 | 白色(255)=放置区域，黑色(0)=不可放置 |

**注意**：plane_image、object_image、mask 必须在数据生成时保持统一尺寸，Dataset 不做 resize。

---

## text_prompt 格式

```
# 推荐：两个 <image> 占位符
"<image>\n<image>\n把椅子放在桌子旁边"

# 纯文本
"把椅子放在桌子旁边"
```

`<image>` 按顺序匹配 `images_path` 中的图片：第一个 = plane_image，第二个 = object_image。

---

## 训练流程

```bash
# Stage 1: Qwen3-VL LoRA 微调（训练完自动提取 seg_features）
python main.py train --config configs/test_stage1.yaml

# Stage 2: Adapter + SAM3 Decoder（读 seg_features，不加载 Qwen3-VL）
python main.py train --config configs/test_stage2.yaml
```

Config 区别：

```yaml
# Stage 1
loss:
  type: "lm"            # 语言模型损失

# Stage 2
loss:
  type: "placement"     # Dice + BCE + rotation + scale
data:
  seg_feature_dir: "data/seg_features/"   # 预提取特征目录
```

---

## 数据规模建议

| 阶段 | 最少 | 推荐 | 关键 |
|------|------|------|------|
| Stage 1 | 100 | 500~1000 | response 质量 > 数量 |
| Stage 2 | 500 | 1000~5000 | mask + rotation_6d + scale 精度 |

推荐划分：训练 80% / 验证 10% / 测试 10%

---

## 快速验证

```python
from src.data.dataset import ObjectPlacementDataset

dataset = ObjectPlacementDataset(
    data_dir="data/",
    ann_file="annotations.json",
    split="train",
    seg_feature_dir="data/seg_features/",  # 可选
)

sample = dataset[0]
print(f"样本数: {len(dataset)}")
print(f"plane_image: {sample['plane_image'].shape}")     # [3, H, W] for SAM3
print(f"images: {len(sample['images'])}")                # number of images for Qwen3-VL
print(f"images[0]: {sample['images'][0].shape}")         # [3, H, W]
print(f"mask: {sample['mask'].shape}")                   # [1, H, W]
print(f"response: {sample['response']}")                 # 含 <SEG>
print(f"rotation_6d: {sample['rotation_6d']}")           # [6] 或 None
print(f"scale: {sample['scale']}")                       # [1] 或 None
print(f"seg_hidden: {sample['seg_hidden'].shape if sample['seg_hidden'] is not None else None}")  # [4096] 或 None
```
