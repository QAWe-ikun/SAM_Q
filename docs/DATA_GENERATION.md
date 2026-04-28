# 训练集自动化生成指南

本文档介绍如何从 SSR3D-FRONT 场景 JSON 直接生成 SAM-Q 训练数据集，无需保存中间 GLB 文件。

---

## 整体流程

```
SSR3D-FRONT 场景 JSON
        ↓
[1] 加载场景，构建完整 3D 房间（含所有家具）
        ↓
[2] 随机选择一个目标物体
        ↓
[3] 渲染原始房间俯视图/侧视图
        ↓
[4] 生成物体参考图（俯视图，正向朝上，缩放随机0.8x-1.5x）
        ↓
[5] 生成 GT 热力图（原位置概率最高，碰撞区域为 0）
        ↓
[6] 从场景剔除目标物体 → 渲染剔除后图像
        ↓
[7] 生成 text_prompt 和 response
        ↓
[8] 保存样本（plane_image / object_image / mask / annotations.json）
        ↓
[可选] 数据增强：随机移动/旋转/缩放物体 → 重复上述流程
```

---

## 数据格式

生成的数据与 `DATASET.md` 格式完全一致：

```
data/
├── train/                          # 训练集
│   ├── scene_001/
│   │   ├── plane_images/
│   │   ├── object_images/
│   │   ├── original_images/        # 原始房间图
│   │   ├── masks/
│   │   ├── debug/
│   │   └── samples.json
│   ├── scene_002/
│   │   └── ...
│   └── ...
├── val/                            # 验证集
└── test/                           # 测试集
```

### samples.json 样本

```json
[
  {
    "sample_id": "scene_001_obj_chair_01",
    "plane_image_path": "plane_images/obj_chair_01.png",
    "object_image_path": "object_images/obj_chair_01.png",
    "mask_path": "masks/obj_chair_01_mask.png",
    "text_prompt": "<image>\n<image>\n把椅子放回原来的位置",
    "response": "好的，我会把椅子放回原来的位置。<SEG>",
    "rotation_6d": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "scale": 1.0,
    "split": "train"
  }
]
```

---

## 核心设计

### 1. 图像输入说明

| 图像类型 | 说明 | 对应字段 |
|---------|------|---------|
| 剔除后房间图 | 剔除目标物体后的房间俯视图或侧视图 | `plane_image_path`, `images_path[1]` |
| 物体参考图 | 目标物体的俯视图，**默认正向朝上，缩放随机** | `images_path[0]` |
| GT 热力图 | 原位置概率最高，四周平滑衰减，碰撞区域概率=0 | `mask_path` |

**重要**：
- VLM 识别物体位置时参考的是**原始房间图**（含所有物体）
- 但输入模型的 `plane_image` 是**剔除后的房间图**
- 这模拟了推理时的真实场景：用户给出指令，模型需要预测物体应该放在哪里

### 2. GT 热力图生成规则

热力图是一个概率分布图，规则如下：

1. **最高概率点**：物体剔除前的中心位置（归一化到图像坐标）
2. **四周衰减**：使用 2D 高斯核从中心向四周平滑衰减，快速降为 0
3. **碰撞区域**：如果某位置与场景中其他物体重叠，该点概率直接设为 0

```python
# 伪代码
heatmap = gaussian_2d(center_x, center_y, sigma=15)  # 中心高斯
collision_mask = compute_collision_mask(scene)
heatmap[collision_mask] = 0.0  # 碰撞区域置零
heatmap = normalize(heatmap)  # 归一化到 [0, 1]
heatmap[heatmap < 0.01] = 0.0  # 阈值处理
```

### 3. 旋转和缩放标签

| 场景 | rotation_6d | scale | 说明 |
|------|-------------|-------|------|
| 基础生成 | 原始旋转的倒数 | 随机缩放的倒数 | 预测如何将物体**恢复到原始状态** |
| 数据增强 | 随机旋转的倒数 | 1.0 | 物体参考图不旋转，预测随机旋转的倒数，以及位置 |

**为什么用倒数？**
- 模型预测的是"如何放置物体"
- 如果物体原始旋转是 30°，模型需要预测 -30° 才能恢复到正向
- 缩放同理：随机缩放 2.0，模型需预测 0.5

### 4. 数据增强流程

```
原始场景
        ↓
[1] 随机选择一个物体
[2] 随机旋转支撑轴（±180°）
[3] 随机移动到场景中不重叠的新位置
[4] 渲染增强后场景图
[5] 剔除该物体 → 渲染剔除后图像
[6] 生成 GT 热力图（基于移动后的位置）
[7] 保存样本
```

**数据增强的标签**：
- `rotation_6d`：随机生成的旋转的倒数（物体参考图不旋转）
- `scale`：1.0（缩放维持为1.0）
- GT 热力图中心点：物体移动后的位置

---

## 使用方法

```bash
# 基础生成（从原始场景剔除物体）
python src/pretreatment/generate_training_data.py \
  --scene_dir d:/3D-Dataset/dataset-ssr3dfront/scenes \
  --model_dir d:/3D-Dataset/3D-FUTURE-model \
  --output_dir data/ \
  --num_samples 1000

# 带数据增强
python src/pretreatment/generate_training_data.py \
  --scene_dir d:/3D-Dataset/dataset-ssr3dfront/scenes \
  --model_dir d:/3D-Dataset/3D-FUTURE-model \
  --output_dir data/ \
  --num_samples 1000 \
  --augmentation \
  --aug_ratio 0.5
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scene_dir` | - | SSR3D-FRONT 场景 JSON 目录 |
| `--model_dir` | - | 3D-FUTURE 模型目录 |
| `--output_dir` | `data/` | 输出数据目录 |
| `--num_samples` | 1000 | 生成样本数量 |
| `--augmentation` | False | 启用数据增强 |
| `--aug_ratio` | 0.5 | 增强样本占总样本的比例 |
| `--image_size` | 1024 | 输出图像分辨率 |
| `--heatmap_sigma` | 15 | 热力图高斯核标准差 |

---

## VLM 集成（可选）

本流程可集成 VLM（如 Qwen3-VL）来识别房间中的物体及其位置：

1. **物体识别**：输入原始房间俯视图，让 VLM 输出所有可见物体的类别和大致位置
2. **位置精炼**：对于选中的目标物体，让 VLM 输出更精确的边界框或中心点

VLM 输出用于：
- 生成 `text_prompt` 中的物体类别
- 验证 GT 热力图中心位置的准确性

---

## 注意事项

1. **碰撞检测**：数据增强时移动物体需确保不与场景中其他物体重叠
2. **物体选择**：优先选择地面（Z≈0）上的物体（支撑面明确）
3. **视角一致性**：俯视图用于地面物体，侧视图用于墙面物体
4. **图像尺寸**：所有输出图像（plane_image, object_image, mask）保持统一尺寸（1024×1024）
5. **比例控制**：基础生成 vs 数据增强的比例建议为 1:1 或 2:1

---

## 快速验证

生成完成后，使用以下代码验证数据格式：

```python
from src.data.dataset import ObjectPlacementDataset

dataset = ObjectPlacementDataset(
    data_dir="data/",
    split_dir="train/",
)

sample = dataset[0]
print(f"plane_image: {sample['plane_image'].shape}")
print(f"mask: {sample['mask'].shape}")
print(f"rotation_6d: {sample['rotation_6d']}")
print(f"scale: {sample['scale']}")
```
