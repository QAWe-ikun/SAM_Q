# 训练集说明

Stage 1 和 Stage 2 的数据集**分开存放**，格式不同。

```
data/
├── stage1/                           # Stage 1 数据（Qwen3-VL LoRA 微调）
│   ├── stage1_annotations.json
│   ├── plane_images/
│   ├── object_images/
│   └── masks/
└── stage2/                           # Stage 2 数据（Adapter + SAM3 Decoder 训练）
    ├── stage2_annotations.json
    ├── plane_images/
    ├── object_images/
    └── masks/
```

---

## Stage 1 数据集

**目标**：教 Qwen3-VL 在正确位置输出 `[SEG]` token

### 目录结构

```
data/stage1/
├── stage1_annotations.json
├── plane_images/
│   ├── scene_001.png
│   └── ...
├── object_images/
│   ├── chair_001.png
│   └── ...
└── masks/                  # Stage 1 也需要 mask（用于验证，Stage 2 继承）
    ├── scene_001_mask.png
    └── ...
```

### stage1_annotations.json 格式

```json
[
  {
    "id": "sample_001",
    "split": "train",
    "plane_image_path": "plane_images/scene_001.png",
    "object_image_path": "object_images/chair_001.png",
    "mask_path": "masks/scene_001_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放在桌子旁边",
    "response": "好的，我会在桌子旁边放置椅子。[SEG]",
    "scene_id": "scene_001",
    "object_id": "chair_001"
  }
]
```

### 字段说明

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `id` | str | 否 | 样本唯一 ID |
| `split` | str | 否 | `"train"` / `"val"` / `"test"` |
| `plane_image_path` | str | 是 | 相对于 `data/stage1/` 的路径 |
| `object_image_path` | str | 是 | 相对于 `data/stage1/` 的路径 |
| `mask_path` | str | 是 | 二值 mask 图路径 |
| `text_prompt` | str | 是 | 放置指令，支持 `<image>` 占位符 |
| `response` | str | **是** | Qwen3-VL 的目标回复，**末尾必须包含 `[SEG]`**。缺失时使用默认值 `"好的，我将为您放置物体。[SEG]"` |
| `scene_id` | str | 否 | 场景标识 |
| `object_id` | str | 否 | 物体标识 |

### response 字段规范

- `[SEG]` 必须出现在回复**末尾**
- 回复应为自然语言描述 + `[SEG]`，不能只有 `[SEG]`
- 语言不限（中英文均可）

```
# 正确示例
"好的，我会在桌子旁边放置椅子。[SEG]"
"I will place the chair next to the table. [SEG]"
"椅子最适合放在窗边，这样采光好。[SEG]"

# 错误示例
"[SEG]"                              # 没有自然语言描述
"好的，我会放置椅子。[SEG] 完成。"   # [SEG] 不在末尾
```

### 数据规模建议

- 最少 **100** 条，推荐 **500~1000** 条
- 关键：**回复质量 > 数量**，确保 `response` 格式正确
- 推荐划分：训练 80% / 验证 10% / 测试 10%

---

## Stage 2 数据集

**目标**：训练 `[SEG]` hidden state → placement heatmap + rotation + scale

### 目录结构

```
data/stage2/
├── stage2_annotations.json
├── plane_images/
│   ├── scene_001.png
│   └── ...
├── object_images/
│   ├── chair_001.png
│   └── ...
└── masks/
    ├── scene_001_mask.png
    └── ...
```

### stage2_annotations.json 格式

```json
[
  {
    "id": "sample_001",
    "split": "train",
    "plane_image_path": "plane_images/scene_001.png",
    "object_image_path": "object_images/chair_001.png",
    "mask_path": "masks/scene_001_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放在桌子旁边",
    "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "scale": 1.0,
    "scene_id": "scene_001",
    "object_id": "chair_001"
  }
]
```

Stage 2 **不需要** `response` 字段（Qwen3-VL 已冻结，只做前向推理）。

### 字段说明

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| `id` | str | 否 | 样本唯一 ID |
| `split` | str | 否 | `"train"` / `"val"` / `"test"` |
| `plane_image_path` | str | 是 | 相对于 `data/stage2/` 的路径 |
| `object_image_path` | str | 是 | 相对于 `data/stage2/` 的路径 |
| `mask_path` | str | 是 | 二值 mask 图路径，**高质量必须** |
| `text_prompt` | str | 是 | 放置指令 |
| `rotation_6d` | list[6] | 是 | 6D 旋转表示（旋转矩阵前两列），参考 Zhou et al. 2019 |
| `scale` | float | 是 | 物体缩放比例，1.0 = 原始大小，>1 放大，<1 缩小 |
| `scene_id` | str | 否 | 场景标识 |
| `object_id` | str | 否 | 物体标识 |

### rotation_6d 说明

6D rotation 是旋转矩阵 R 的前两列向量拼接：`[r11, r21, r31, r12, r22, r32]`

```
旋转矩阵 R:
| r11  r12  r13 |
| r21  r22  r23 |
| r31  r32  r33 |

→ rotation_6d = [r11, r21, r31, r12, r22, r32]
```

常见值：
- **无旋转（单位矩阵）**: `[1, 0, 0, 0, 1, 0]`
- **绕 Z 轴旋转 90°**: `[0, 1, 0, -1, 0, 0]`
- **绕 Z 轴旋转 45°**: `[0.707, 0.707, 0, -0.707, 0.707, 0]`

### 数据规模建议

- 最少 **500** 条，推荐 **1000~5000** 条
- 关键：**mask 质量**，白色区域必须精确标注放置位置
- 推荐划分：训练 80% / 验证 10% / 测试 10%

---

## 图像要求

### plane_image（房间/平面俯视图）

- **格式**：PNG / JPG，RGB
- **推荐分辨率**：1024×1024（DataLoader 自动 resize）
- **内容**：房间或桌面的**俯视图**，视角垂直向下

### object_image（物体俯视图）

- **格式**：PNG / JPG，RGB
- **推荐分辨率**：1024×1024
- **内容**：待放置物体的**俯视图**，背景尽量干净

### mask（放置 mask）

- **格式**：PNG，灰度图（L 模式）
- **像素值**：白色（255）= 可放置区域，黑色（0）= 不可放置
- **分辨率**：与 plane_image 一致（DataLoader 自动 resize）

---

## text_prompt 格式

```
# 推荐：两个 <image> 占位符，第一个对应 plane_image，第二个对应 object_image
"<image>\n<image>\n把椅子放在桌子旁边"

# 纯文本（不含图片描述）
"把椅子放在桌子旁边"
```

---

## Config 配置

不同阶段通过 config 的 `data.root_dir` 和 `data.ann_file` 区分：

```yaml
# configs/stage1_qwen_lora.yaml
data:
  root_dir: "data/stage1/"
  ann_file: "stage1_annotations.json"

# configs/stage2_decoder.yaml
data:
  root_dir: "data/stage2/"
  ann_file: "stage2_annotations.json"
```

---

## 快速验证

```python
from src.data.dataset import ObjectPlacementDataset

# Stage 1
ds1 = ObjectPlacementDataset(
    data_dir="data/stage1/",
    ann_file="stage1_annotations.json",
    split="train",
)
sample = ds1[0]
print(f"Stage1 样本数: {len(ds1)}")
print(f"response: {sample['response']}")  # 应包含 [SEG]

# Stage 2
ds2 = ObjectPlacementDataset(
    data_dir="data/stage2/",
    ann_file="stage2_annotations.json",
    split="train",
)
print(f"Stage2 样本数: {len(ds2)}")
print(f"mask shape: {ds2[0]['mask'].shape}")  # [1, H, W]
```
