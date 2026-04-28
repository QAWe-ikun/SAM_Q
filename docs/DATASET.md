# 训练集说明

Stage 1 和 Stage 2 **共用同一个数据集**，统一放在 `data/` 下。
Stage 1 训练结束后自动提取 `<SEG>` hidden states 到 `data/seg_features/`，Stage 2 直接读取，无需加载 Qwen3-VL。

## 目录结构

```
data/
├── train/                          # 训练集
│   ├── train.json                  # 所有训练样本的元数据（合并）
│   ├── scene_001/                  # 一个原始场景对应一个文件夹
│   │   ├── plane_images/           # 剔除不同物体后的房间俯视图
│   │   │   ├── obj_chair_01.png    # 剔除椅子后的房间图
│   │   │   ├── obj_table_02.png    # 剔除桌子后的房间图
│   │   │   └── ...
│   │   ├── object_images/          # 被剔除物体的参考图
│   │   │   ├── obj_chair_01.png    # 椅子参考图
│   │   │   ├── obj_table_02.png    # 桌子参考图
│   │   │   └── ...
│   │   ├── original_images/        # 包含所有物体的原始房间图（用于调试/参考）
│   │   │   ├── obj_chair_01.png    # 原始房间图（含椅子）
│   │   │   ├── obj_table_02.png    # 原始房间图（含桌子）
│   │   │   └── ...
│   │   └── masks/                  # GT 放置热力图
│   │       ├── obj_chair_01_mask.png
│   │       ├── obj_table_02_mask.png
│   │       └── ...
│   ├── scene_002/
│   │   ├── plane_images/
│   │   ├── object_images/
│   │   ├── original_images/
│   │   └── masks/
│   └── ...
├── val/                            # 验证集（结构同 train/）
│   └── val.json
├── test/                           # 测试集（结构同 train/）
│   └── test.json
└── seg_features/                   # <SEG> hidden states（Stage 1 训练后自动生成）
    ├── scene_001_obj_chair_01.pt
    └── ...
```

每个 `{split}.json` 包含该 split 下所有场景的样本元数据：

```json
[
  {
    "sample_id": "scene_001_obj_chair_01",
    "scene_dir": "scene_001",
    "plane_image_path": "plane_images/obj_chair_01.png",
    "images_paths": [
      "object_images/obj_chair_01.png",
      "plane_images/obj_chair_01.png"
    ],
    "mask_path": "masks/obj_chair_01_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放回原来的位置",
    "response": "好的，我会把椅子放回原来的位置，旋转0.0°，缩放1.00倍。<SEG>",
    "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "scale": 1.0,
    "split": "train"
  },
  {
    "sample_id": "scene_001_obj_table_02",
    "scene_dir": "scene_001",
    "plane_image_path": "plane_images/obj_table_02.png",
    "images_paths": [
      "object_images/obj_table_02.png",
      "plane_images/obj_table_02.png"
    ],
    "mask_path": "masks/obj_table_02_mask.png",
    "text_prompt": "<image>\n<image>\n把桌子放回原来的位置",
    "response": "好的，我会把桌子放回原来的位置，旋转90.0°，缩放0.80倍。<SEG>",
    "rotation_6d": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
    "scale": 0.8,
    "split": "train"
  }
]
```

---

## 字段说明

| 字段 | 类型 | Stage 1 | Stage 2 | 说明 |
|------|------|---------|---------|------|
| `sample_id` | str | 必须 | 必须 | 样本唯一标识符，格式 `{scene_id}_obj_{obj_type}_{idx}` |
| `scene_dir` | str | 必须 | 必须 | 场景文件夹名称（相对于 split 目录） |
| `plane_image_path` | str | 必须 | 必须 | SAM3 输入（房间/平面俯视图），相对于 scene 文件夹 |
| `images_paths` | list[str] | 必须 | 必须 | 所有输入图片列表（通常为 plane + object），相对于 scene 文件夹 |
| `mask_path` | str | 必须 | 必须 | 二值 mask 图，白色 = 放置区域，相对于 scene 文件夹 |
| `text_prompt` | str | 必须 | 必须 | 放置指令，支持 `<image>` 占位符 |
| `response` | str | **必须** | 不需要 | Qwen3-VL 的目标回复，末尾包含 `<SEG>`，硬编码了旋转角度和缩放 |
| `rotation_6d` | list[6] | 不需要 | **必须** | 6D 旋转表示（旋转矩阵前两列） |
| `scale` | float | 不需要 | **必须** | 缩放比例，1.0 = 原始大小 |
| `split` | str | 可选 | 可选 | `"train"` / `"val"` / `"test"` |

### images_paths 设计原则

```json
{
  "plane_image_path": "plane_images/obj_chair_01.png",       // SAM3 专用
  "images_paths": [                                           // Qwen3-VL 多模态输入
    "object_images/obj_chair_01.png"                           // 第1张 → 第1个 <image>
    "plane_images/obj_chair_01.png",                           // 第2张 → 第2个 <image>
  ]
}
```

- `plane_image_path` 单独列出，因为 SAM3 只处理房间/平面俯视图
- `images_paths` 包含所有要传给 Qwen3-VL 的图片，**顺序必须与 `text_prompt` 中 `<image>` 的出现顺序一致**
- 通常 `images_paths[0] == plane_image_path`，`images_paths[1] == object_image_path`

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
- **无旋转**: `[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]`
- **绕 Z 轴旋转 90°**: `[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]`
- **绕 Z 轴旋转 45°**: `[0.707, 0.707, 0.0, -0.707, 0.707, 0.0]`

---

## scale 说明（Stage 2）

`scale` 是相对于物体原始尺寸的缩放比例：

- **1.0** = 保持原始大小
- **0.5** = 缩小到 50%
- **2.0** = 放大到 200%

建议范围：`0.8 ~ 1.25`

---

## seg_features 目录

Stage 1 训练结束后**自动生成**，无需手动创建。

每个 `.pt` 文件包含：
```python
{
    "seg_hidden": torch.Tensor,  # [4096]  Qwen3-VL 在 <SEG> 位置的 hidden state
    "sample_id": str,            # 对应 samples.json 中的 sample_id
}
```

文件名格式：`{sample_id}.pt`（与 samples.json 的 `sample_id` 对应）

---

## 图像要求

| 类型 | 格式 | 推荐分辨率 | 说明 |
|------|------|-----------|------|
| plane_image | PNG/JPG, RGB | 1024×1024 | 房间或桌面的俯视图 |
| object_image | PNG/JPG, RGB | 1024×1024 | 物体的俯视图，背景尽量干净 |
| mask | PNG, 灰度 | 与 plane_image 一致 | 概率图：0=不可放置，255=最佳位置 |

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
python main.py train --config configs/stage1_qwen_lora.yaml

# Stage 2: Adapter + SAM3 Decoder（读 seg_features，不加载 Qwen3-VL）
python main.py train --config configs/stage2_decoder.yaml
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
    split_dir="train/",
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
