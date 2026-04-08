# 配置文件迁移指南

本文档说明如何从旧配置系统迁移到新的配置系统。

---

## 配置系统重构

### 旧配置系统（已删除）

```
configs/
├── config.yaml                          # 基础配置
├── sam2qhmvpl_config.yaml               # H-MVP配置
├── sam2qvla_incremental_config.yaml     # 增量VLA配置
└── vla_config.yaml                      # VLA配置
```

**问题**：
- ❌ 配置之间重复严重
- ❌ 无继承机制，修改麻烦
- ❌ 命名不规范
- ❌ 路径硬编码

### 新配置系统（当前）

```
configs/
├── base.yaml              # 基础配置（所有默认值）
├── hmvp.yaml              # H-MVP扩展（继承base.yaml）
└── incremental_vla.yaml   # VLA扩展（继承hmvp.yaml）
```

**优势**：
- ✅ 配置继承，避免重复
- ✅ 层次清晰，易于维护
- ✅ 命名规范，语义明确
- ✅ 路径相对化，可移植

---

## 配置映射关系

| 旧配置 | 新配置 | 说明 |
|--------|--------|------|
| `config.yaml` | `base.yaml` | 基础配置，包含所有默认值 |
| `sam2qhmvpl_config.yaml` | `hmvp.yaml` | H-MVP碰撞检测配置 |
| `sam2qvla_incremental_config.yaml` | `incremental_vla.yaml` | 增量VLA记忆配置 |
| `vla_config.yaml` | 已整合 | 功能分散到 `base.yaml` 和 `hmvp.yaml` |

---

## 配置对比示例

### 1. 基础配置

#### 旧配置 (config.yaml)
```yaml
data_dir: "D:/experiment/SAM_Q/data"
batch_size: 4
num_workers: 4

model:
  qwen_model_name: "Qwen/Qwen3-VL-7B-Instruct"
  sam3_input_dim: 256
  qwen_hidden_dim: 3584
  adapter_hidden_dim: 512

freeze_qwen: true
freeze_sam3_image_encoder: true

optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4

num_epochs: 100
output_dir: "D:/experiment/SAM_Q/outputs"
```

#### 新配置 (base.yaml)
```yaml
experiment:
  name: "samq_base"
  seed: 42

data:
  root_dir: "data/"                    # 相对路径
  batch_size: 4
  num_workers: 4
  plane_image_size: [1024, 1024]
  object_image_size: [512, 512]

model:
  qwen:
    model_name: "Qwen/Qwen3-VL-7B-Instruct"
    hidden_dim: 3584
    freeze: true
  sam3:
    input_dim: 256
    freeze_image_encoder: true
  adapter:
    type: "cross_modal"
    hidden_dim: 512
    num_queries: 64

loss:
  type: "placement"
  dice_weight: 1.0
  bce_weight: 1.0

optimizer:
  type: "AdamW"
  lr: 1.0e-4
  weight_decay: 1.0e-4

scheduler:
  type: "CosineAnnealingLR"
  T_max: 100
  eta_min: 1.0e-6
  warmup_epochs: 5

training:
  num_epochs: 100
  save_dir: "outputs/"
  save_interval: 10
  save_best: true
```

**改进点**：
- ✅ 层次化组织（data、model、optimizer等分组）
- ✅ 添加实验元数据（name、seed）
- ✅ 更完整的配置项（scheduler、warmup等）
- ✅ 使用相对路径

---

### 2. H-MVP配置

#### 旧配置 (sam2qhmvpl_config.yaml)
```yaml
data_dir: "D:/experiment/SAM_Q/data"
batch_size: 2
num_workers: 2

model:
  sam_high_res: 1024
  sam_low_res: 256
  hmvp_max_level: 4
  hmvp_base_resolution: 8
  hmvp_early_out_threshold: 0.1
  lifting_hidden_dim: 256
  num_candidates: 5
  optimization_steps: 10
  freeze_sam2_components: true
  freeze_qwen_components: true

mask_weight: 1.0
text_weight: 0.5

num_epochs: 50
output_dir: "D:/experiment/SAM_Q/outputs_sam2qhmvpl"
```

#### 新配置 (hmvp.yaml)
```yaml
_base_: "base.yaml"                    # 继承基础配置

experiment:
  name: "samq_hmvp"
  description: "SAM-Q with H-MVP collision detection"

model:
  sam3:
    freeze_image_encoder: true
    freeze_detector: false
  adapter:
    type: "cross_modal"
    hidden_dim: 512
    num_queries: 64

advanced:
  dual_scale:
    enabled: true
    high_res: 1024
    low_res: 256
  hmvp:
    enabled: true
    max_level: 4
    base_resolution: 8
    early_out_threshold: 0.1
  incremental_vla:
    enabled: false

training:
  num_epochs: 50
  batch_size: 2
  save_dir: "outputs_hmvp/"

loss:
  type: "vla"
  heatmap_weight: 1.0
  collision_weight: 0.5
  semantic_weight: 0.3
```

**改进点**：
- ✅ 只需指定与base.yaml的差异部分
- ✅ 高级模块统一在 `advanced` 下管理
- ✅ 配置结构更清晰

---

## 使用方式对比

### 旧使用方式
```bash
# 训练基础模型
python main.py train --config configs/config.yaml

# 训练H-MVP模型
python main.py train --config configs/sam2qhmvpl_config.yaml
```

### 新使用方式
```bash
# 训练基础模型
python main.py train --config configs/base.yaml

# 训练H-MVP模型
python main.py train --config configs/hmvp.yaml

# 训练增量VLA模型
python main.py train --config configs/incremental_vla.yaml
```

---

## 程序化访问对比

### 旧方式
```python
import yaml

# 需要手动加载
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 访问嵌套值容易出错
lr = config["optimizer"]["lr"]  # 可能 KeyError
```

### 新方式
```python
from src.utils.config import Config

config = Config("configs/hmvp.yaml")

# 点符号访问，支持默认值
lr = config.get("optimizer.lr", 1e-4)
hmvp_enabled = config.get("advanced.hmvp.enabled", False)

# 修改配置
config.set("training.num_epochs", 200)

# 保存配置
config.save("outputs/my_config.yaml")
```

---

## 配置项完整映射

### 数据配置
| 旧键 | 新键 |
|------|------|
| `data_dir` | `data.root_dir` |
| `batch_size` | `data.batch_size` |
| `num_workers` | `data.num_workers` |
| - | `data.plane_image_size` |
| - | `data.object_image_size` |

### 模型配置
| 旧键 | 新键 |
|------|------|
| `model.qwen_model_name` | `model.qwen.model_name` |
| `model.qwen_hidden_dim` | `model.qwen.hidden_dim` |
| `model.sam3_input_dim` | `model.sam3.input_dim` |
| `model.adapter_hidden_dim` | `model.adapter.hidden_dim` |
| `freeze_qwen` | `model.qwen.freeze` |
| `freeze_sam3_image_encoder` | `model.sam3.freeze_image_encoder` |

### 训练配置
| 旧键 | 新键 |
|------|------|
| `num_epochs` | `training.num_epochs` |
| `output_dir` | `training.save_dir` |
| - | `training.save_interval` |
| - | `training.save_best` |
| - | `training.early_stopping` |
| - | `training.patience` |

### 优化器配置
| 旧键 | 新键 |
|------|------|
| `optimizer.lr` | `optimizer.lr` |
| `optimizer.weight_decay` | `optimizer.weight_decay` |
| - | `optimizer.type` |
| - | `optimizer.betas` |

---

## 迁移检查清单

如果你有旧的训练脚本或检查点，请按以下步骤迁移：

- [ ] 更新配置文件路径：`config.yaml` → `base.yaml`
- [ ] 更新配置键访问：使用新的层次结构
- [ ] 检查点兼容：旧检查点中的保存的config可能不兼容新结构
- [ ] 测试训练流程：确保新配置系统工作正常
- [ ] 更新文档：确保所有示例使用新配置

---

## 常见问题

### Q: 旧检查点还能用吗？
A: 可以，但需要手动处理配置不兼容问题。加载时：
```python
checkpoint = torch.load("old_checkpoint.pt")
old_config = checkpoint["config"]
# 手动转换为新格式
new_config = convert_old_config(old_config)
```

### Q: 如何创建自定义配置？
A: 继承base.yaml：
```yaml
_base_: "base.yaml"

experiment:
  name: "my_experiment"

training:
  num_epochs: 200
  save_dir: "outputs/my_exp/"
```

### Q: 可以在运行时覆盖配置吗？
A: 可以：
```python
config = Config("configs/base.yaml")
config.set("training.num_epochs", 150)
config.set("data.batch_size", 8)
```

---

## 总结

新配置系统提供了：
- ✅ **继承机制**：避免重复
- ✅ **层次结构**：清晰组织
- ✅ **类型安全**：点符号访问
- ✅ **易于扩展**：添加新配置项简单
- ✅ **可移植性**：使用相对路径

**迁移完成日期**: 2026-04-08  
**删除的旧配置**: 4个  
**新增的新配置**: 3个
