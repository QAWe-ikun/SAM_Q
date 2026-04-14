# SAM-Q Architecture Documentation

This document provides a comprehensive overview of the SAM-Q system architecture, design decisions, and implementation details.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Principles](#design-principles)
3. [Architecture Diagram](#architecture-diagram)
4. [Module Details](#module-details)
5. [Data Flow](#data-flow)
6. [Configuration System](#configuration-system)
7. [Extension Guide](#extension-guide)

---

## System Overview

SAM-Q is a modular framework for **semantically-aware object placement** in indoor scenes. It combines:

- **SAM3** (Segment Anything Model 3): For spatial understanding and mask generation
- **Qwen3-VL-8B** (Vision-Language Model): For multimodal comprehension (images + text)
- **Novel Adapter Architecture**: For cross-modal embedding alignment
- **H-MVP** (Hierarchical Multi-View Projection): For 3D collision detection
- **Incremental VLA Memory**: For dynamic scene understanding

### Key Capabilities

| Capability | Description | Module |
|------------|-------------|--------|
| Language-Guided Placement | Natural language controls placement semantics | `models/encoders/` |
| Cross-Modal Fusion | Bridges 4096D (Qwen) to 256D (SAM3) | `models/adapters/` |
| 3D Collision Detection | Prevents invalid placements | `models/collision/` |
| Incremental Memory | Updates scene understanding | `models/vla/` |
| Heatmap-Guided Sampling | Multiple candidate extraction | `models/sampling/` |

---

## Design Principles

### 1. Modularity

Each component is independent and interchangeable:

```
Encoder -> Adapter -> Detector -> Sampling
   |          |         |          |
 Qwen3-VL  Cross-Attn  SAM3     Top-K+NMS
```

### 2. Configuration Inheritance

Configs inherit from base configs, avoiding duplication:

```yaml
# hmvp.yaml
_base_: "base.yaml"  # Inherit all base settings

advanced:
  hmvp:
    enabled: true    # Override only what's needed
```

### 3. Lazy Loading

Heavy models (Qwen3-VL, SAM3) are loaded lazily to:
- Reduce import time
- Enable development without GPU
- Support testing with mocks

### 4. Parameter Efficiency

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| Qwen3-VL-8B | 8B | No (frozen) |
| SAM3 Image Encoder | ~100M | No (frozen) |
| SAM3 Detector | ~10M | Yes |
| Cross-Modal Adapter | ~5M | Yes |
| **Total** | **~8.1B** | **<5%** |

---

## Architecture Diagram

```
+-------------------------------------------------------------------+
|                          INPUT LAYER                               |
|                                                                    |
|  Room Image (1024x1024)    Object Image (512x512)    Text Prompt   |
|         |                          |                        |      |
+---------|--------------------------|------------------------|------+
          |                          |                        |
          v                          v                        v
+-------------------------------------------------------------------+
|                      ENCODER LAYER (Frozen)                        |
|                                                                    |
|  +---------------------+        +----------------------------+     |
|  | SAM3 Image Encoder  |        | Qwen3-VL Vision-Language   |     |
|  | Output: 256D/patch  |        | Output: 4096D/token        |     |
|  +----------+----------+        +-------------+--------------+     |
|             |                                |                     |
|             |                       +--------v---------+          |
|             |                       | Adapter Module   |          |
|             |                       | 4096D -> 256D    |          |
|             |                       | + Cross-Attn     |          |
|             |                       +------------------+          |
+-------------+------------------------------+----------------------+
              |                              |
              v                              v
+-------------------------------------------------------------------+
|                     FUSION & DECODER LAYER                         |
|                                                                    |
|  +------------------------------------------------------------+    |
|  |  Plane & Text Embeddings -> SAM3 Detector -> Placement Masks|   |
|  +------------------------------------------------------------+    |
+-------------------------------------------------------------------+
              |
              v
+-------------------------------------------------------------------+
|                   ADVANCED MODULES (Optional)                      |
|                                                                    |
|  +--------------------------+     +------------------------+       |
|  | H-MVP Collision Detector |     | Incremental VLA System |       |
|  +--------------------------+     +------------------------+       |
+-------------------------------------------------------------------+
```

---

## Module Details

### 1. Encoders (`src/models/encoders/`)

#### Qwen3VLEncoder

**Purpose**: Replace SAM3's text encoder with multimodal vision-language capabilities.

**Interface**:
```python
encoder = Qwen3VLEncoder(model_name="Qwen/Qwen3-VL-8B-Instruct")
embeddings = encoder(object_image=pil_img, text_prompt="Place near window")
# Output: (batch, seq_len, 4096)
```

**Key Features**:
- Lazy model loading
- Flash attention support (CUDA)
- Conversation-format input

**[SEG] Token Bridging (SA2VA-style)**:

The encoder supports a special `[SEG]` token mode that bridges Qwen3-VL reasoning to SAM3 segmentation:

```python
# Single SEG (most common): [B, 4096]
seg_hidden, _ = encoder.generate_with_seg(obj_img, "放一把椅子", force_only=True, num_seg=1)

# Multi SEG (advanced): [B, num_seg, 4096]
seg_hidden, _ = encoder.generate_with_seg(obj_img, "放椅子和桌子", force_only=True, num_seg=8)
```

How it works:
```
Input: [Image Tokens] [Text Tokens] [SEG]
       ↓
Self-Attention (causal mask): [SEG] attends to all preceding tokens
       ↓
[SEG] Hidden State = attend(image_tokens + text_tokens)
       → NOT a fixed vector; dynamically computed from context
       ↓
SEG Projector → SAM3 Decoder → Placement Mask
```

Key design decisions:
- `[SEG0]`~`[SEG63]` are added to the vocabulary as special tokens
- `force_only=True` (training): directly append [SEG] and extract hidden state in one pass
- `force_only=False` (inference): model may naturally generate [SEG]; fallback to force-append
- Hidden states vary with input — the model learns *when* and *where* to trigger segmentation

**LoRA/QLoRA Fine-Tuning**:

```python
encoder.enable_finetuning(
    lora_r=64, lora_alpha=128, use_qlora=False,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
)
# Trainable: ~8M / 8B (0.10%)
```

LoRA is applied to 7 linear layers per Transformer block:
| Layer | Role in SAM-Q |
|-------|--------------|
| `q_proj` | Learn to attend to relevant image regions based on text |
| `k_proj` | Make image tokens better match text semantics |
| `v_proj` | Pass correct visual features for placement |
| `o_proj` | Integrate multi-modal information |
| `gate_proj` | Activate task-specific features |
| `up_proj` | Expand representation space for placement |
| `down_proj` | Compress task-specific information |

**Implementation**: `src/models/encoders/qwen3vl_encoder.py`

---

### 2. Adapters (`src/models/adapters/`)

#### Base Adapter

Simple MLP for dimension reduction:

```
Input (4096D) -> Linear(512) -> LayerNorm -> GELU -> Dropout -> Linear(256D)
```

#### CrossModalAdapter (Core)

Cross-attention based adapter:

```python
adapter = CrossModalAdapter(
    qwen_dim=4096,
    sam3_dim=256,
    num_queries=64,
    hidden_dim=512,
)
output = adapter(qwen_embeddings)
# Output: (batch, 64, 256) - Fixed length queries
```

**Architecture**:
1. Input projection: `4096D -> 512D`
2. Learnable queries: `64 x 512D`
3. Cross-attention: Queries attend to input
4. Output projection: `512D -> 256D`

**Why 64 queries?**
- Matches DETR-style detector expectations
- Balances expressiveness vs. computation
- Empirically sufficient for placement tasks

#### PresenceTokenAdapter

Adds learnable presence tokens to help distinguish similar prompts.

---

### 3. Collision Detection (`src/models/collision/`)

#### HMVPCollisionDetector

**Concept**: Hierarchical Multi-View Projection for 3D collision detection.

**How it works**:
1. Build depth pyramid (4 levels) for scene and object
2. Project onto 6 orthogonal views per level
3. Check depth interval overlap
4. Early exit if confident separation

**Interface**:
```python
detector = HMVPCollisionDetector(
    max_level=4,
    base_resolution=8,
    early_out_threshold=0.1,
)

collision_score = detector.check_collision(
    scene_depths=scene_hmvp,
    object_depths=obj_hmvp,
    pose=predicted_pose,
)
```

**Differentiable Operations**:
- `soft_depth_interval_overlap`: Smooth overlap calculation
- `soft_max_pool2d`: Softmax-based max approximation

---

### 4. VLA Memory (`src/models/vla/`)

#### IncrementalHMVPMemory

**Innovation**: Dynamic scene understanding that updates with each placement.

**Workflow**:
```
Initial Scene -> Build H-MVP
      |
Place Object A
      |
Update H-MVP (incremental, not rebuild)
      |
Place Object B
      |
Update H-MVP
      |
...
```

**Key Methods**:
- `initialize_from_scene()`: Build initial H-MVP
- `update_with_new_object()`: Incremental update
- `update_with_object_movement()`: Handle object movement

---

### 5. Sampling (`src/models/sampling/`)

#### HeatmapGuidedPlacer

**Purpose**: Extract diverse placement candidates from probability heatmap.

**Pipeline**:
1. Refine heatmap with SAM features
2. Top-K peak extraction
3. 2D NMS for diversity
4. Convert 2D locations to 3D poses

**Interface**:
```python
placer = HeatmapGuidedPlacer()
candidates = placer.extract(
    heatmap=probability_map,
    num_candidates=5,
    nms_radius=0.1,
)
```

---

## Data Flow

### Training

```
annotations.json
      |
ObjectPlacementDataModule
      |
+----------------------------+
| DataLoader (batch=4)       |
|  - plane_images: List[PIL] |
|  - object_images: List[PIL]|
|  - text_prompts: List[str] |
|  - masks: Tensor[B, 1, H, W]|
+------------|---------------+
             v
SAMQPlacementModel.forward()
             |
+----------------------------+
| For each sample in batch:  |
|  1. plane_image -> SAM3 encoder  |
|  2. object+text -> Qwen -> Adapter|
|  3. Combined -> SAM3 detector    |
|  4. Output: masks               |
+------------|---------------+
             v
PlacementLoss(masks, targets)
             |
+----------------------------+
| Loss Components:           |
|  - Dice Loss (weight=1.0)  |
|  - BCE Loss (weight=1.0)   |
+------------|---------------+
             v
backward() -> optimizer.step()
```

### Inference

```
plane_image + object_image + text
             |
PlacementPredictor.predict()
             |
+----------------------------+
| 1. Resize images           |
| 2. Model forward (no grad) |
| 3. Apply threshold         |
| 4. Extract boxes & scores  |
| 5. Create heatmap visualization|
+------------|---------------+
             v
Results: {mask, heatmap, boxes, scores}
```

---

## Configuration System

### Hierarchy

```
base.yaml (all defaults)
    |
hmvp.yaml (enable H-MVP)
    |
incremental_vla.yaml (enable incremental VLA)
```

### Key Sections

| Section | Purpose | Example Keys |
|---------|---------|--------------|
| `experiment` | Metadata | name, seed |
| `data` | Dataset settings | batch_size, image sizes |
| `model` | Architecture | qwen, sam3, adapter |
| `loss` | Loss weights | dice_weight, bce_weight |
| `optimizer` | Optimizer | lr, weight_decay |
| `scheduler` | LR scheduling | T_max, eta_min |
| `training` | Training loop | num_epochs, save_dir |
| `advanced` | Optional modules | hmvp |

### Usage

```python
from src.utils.config import Config

config = Config("configs/hmvp.yaml")

# Access with dot notation
lr = config.get("optimizer.lr")
hmvp_enabled = config.get("advanced.hmvp.enabled")

# Override
config.set("training.num_epochs", 200)

# Save
config.save("outputs/my_config.yaml")
```

---

## Extension Guide

### Adding a New Adapter

1. Create file: `src/models/adapters/my_adapter.py`
2. Implement:
```python
import torch.nn as nn

class MyAdapter(nn.Module):
    def __init__(self, qwen_dim, sam3_dim, ...):
        super().__init__()
        # Initialize layers
    
    def forward(self, x):
        # Transform x
        return output
```
3. Add to `src/models/adapters/__init__.py`
4. Update config: `model.adapter.type: "my_adapter"`

### Adding a New Loss

1. Create file: `src/models/losses/my_loss.py`
2. Implement:
```python
class MyLoss(nn.Module):
    def __init__(self, weight=1.0):
        self.weight = weight
    
    def forward(self, pred, target):
        # Compute loss
        return loss_value
```
3. Integrate into trainer

### Adding a New Encoder

1. Create file: `src/models/encoders/my_encoder.py`
2. Must implement:
   - `forward()` method
   - `output_dim` property
3. Register in `__init__.py`

---

## File Structure

```
SAM-Q/
|-- configs/                     # Configuration files
|   |-- base.yaml               # Base configuration
|   |-- hmvp.yaml               # H-MVP extension
|   +-- incremental_vla.yaml    # VLA extension
|
|-- src/
|   |-- models/                  # Model architectures
|   |   |-- encoders/           # Encoder modules
|   |   |   +-- qwen3vl_encoder.py
|   |   |-- adapters/           # Adapter modules
|   |   |   |-- base_adapter.py
|   |   |   |-- cross_modal_adapter.py
|   |   |   +-- presence_token_adapter.py
|   |   |-- collision/          # Collision detection
|   |   |-- vla/                # VLA components
|   |   |-- sampling/           # Sampling strategies
|   |   +-- placement_model.py  # Main model
|   |
|   |-- data/                    # Data pipeline
|   |   |-- dataset.py
|   |   |-- vla_dataset.py
|   |   +-- transforms.py
|   |
|   |-- train/                   # Training framework
|   |   |-- trainer.py
|   |   |-- optimizer.py
|   |   +-- metrics.py
|   |
|   |-- inference/               # Inference utilities
|   |   |-- predictor.py
|   |   +-- visualizer.py
|   |
|   +-- utils/                   # Utilities
|       +-- config.py
|
|-- tests/                       # Unit tests
|   |-- test_models/
|   +-- test_data/
|
|-- scripts/                     # Helper scripts
|   |-- download_data.sh
|   +-- evaluate.py
|
|-- main.py                      # CLI entry point
|-- README.md                    # Documentation
+-- ARCHITECTURE.md              # This file
```

---

## Performance Optimization

### Memory Efficiency

1. **Freeze foundation models**: Qwen3-VL and SAM3 encoder frozen
2. **Gradient checkpointing**: Enable for long sequences
3. **Mixed precision**: Use `torch.float16` on supported GPUs
4. **Flash Attention 2**: Reduces activation memory by ~30-40%
5. **LoRA/QLoRA**: Only trainable parameters stored in optimizer state

### VRAM Breakdown (Qwen3-VL-8B + SAM3)

| Component | Without FA2 | With FA2 | QLoRA 4-bit + FA2 |
|-----------|-------------|----------|--------------------|
| Model weights | ~17 GB | ~17 GB | ~6 GB |
| Activations + KV cache | ~8 GB | ~4 GB | ~3 GB |
| SAM3 | ~1.5 GB | ~1.5 GB | ~1.5 GB |
| **Total** | **~26.5 GB** | **~22.5 GB** | **~10.5 GB** |

### Speed Optimization

1. **Flash Attention**: Set `attn_implementation: "flash_attention_2"`
2. **Batch size**: Maximize within VRAM limits
3. **Workers**: Set `num_workers` to CPU cores / 2

### Scaling

| GPU VRAM | Recommended batch_size | Mode |
|----------|----------------------|------|
| 8GB | 1 | QLoRA 4-bit |
| 16GB | 2-4 | QLoRA 4-bit / FP16 inference |
| 24GB (RTX 4090) | 1 (FP16 train) / 4 (QLoRA train) | FP16 + FA2 / QLoRA 4-bit |
| 40GB+ | 16+ | Full precision |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce batch_size, enable gradient checkpointing |
| Slow imports | Verify lazy loading is enabled |
| Dimension mismatch | Check adapter hidden_dim matches config |
| CUDA errors | Verify device consistency across components |

---

## Future Work

- [ ] Support for multiple object placement
- [ ] Real-time interactive demo
- [ ] Integration with 3D asset databases
- [ ] Multi-GPU distributed training
- [ ] ONNX export for deployment

---

## Contributing

See [README.md](README.md) for contribution guidelines.

---

**Version**: 1.0.0  
**Last Updated**: 2026-04-09
