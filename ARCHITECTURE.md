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

### Key Capabilities

| Capability | Description | Module |
|------------|-------------|--------|
| Language-Guided Placement | Natural language controls placement semantics | `models/encoders/` |
| Cross-Modal Fusion | Bridges 4096D (Qwen) to 256D (SAM3) | `models/adapters/` |

---

## Design Principles

### 1. Modularity

Each component is independent and interchangeable:

```
Encoder -> Adapter -> Detector -> Sampling
   |          |         |          |
 Qwen3-VL  Cross-Attn  SAM3     Top-K+NMS
```

### 2. Lazy Loading

Heavy models (Qwen3-VL, SAM3) are loaded lazily to:
- Reduce import time
- Enable development without GPU
- Support testing with mocks

### 3. Parameter Efficiency

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
|  | Plane & Text Embeddings -> SAM3 Detector -> Placement Masks|    |
|  +------------------------------------------------------------+    |
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

**<SEG> Token Bridging (SA2VA-style)**:

The encoder supports a special `<SEG>` token mode that bridges Qwen3-VL reasoning to SAM3 segmentation:

```python
# Single SEG (most common): [B, 4096]
seg_hidden, _ = encoder.generate_with_seg(obj_img, "放一把椅子", force_only=True, num_seg=1)

# Multi SEG (advanced): [B, num_seg, 4096]
seg_hidden, _ = encoder.generate_with_seg(obj_img, "放椅子和桌子", force_only=True, num_seg=8)
```

How it works:
```
Input: [Image Tokens] [Text Tokens] <SEG>
       ↓
Self-Attention (causal mask): <SEG> attends to all preceding tokens
       ↓
<SEG> Hidden State = attend(image_tokens + text_tokens)
       → NOT a fixed vector; dynamically computed from context
       ↓
SEG Projector → SAM3 Decoder → Placement Mask
```

Key design decisions:
- `<SEG>` is added to the vocabulary as a special token
- `force_only=True` (training): directly append <SEG> and extract hidden state in one pass
- `force_only=False` (inference): model may naturally generate <SEG>; fallback to force-append
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

### 3. VLA Module (`src/models/vla/`)

#### Unified Architecture (Parallel SAM3 + SEGActionHead)

**Core Idea**: Single `<SEG>` token feeds two parallel branches:
- **SAM3 Decoder**: Generates placement heatmap (2D position)
- **SEGActionHead**: Generates rotation angle + relative scale

```
Qwen3-VL → <SEG> hidden state [B, 4096]
    ↓
    ├→ CrossModalAdapter → SAM3 Decoder → Heatmap [B, num_candidates, H, W]
    └→ SEGActionHead → rotation [B] + scale [B]
```

**Output from `forward()`**:
```python
{
    "heatmap": [B, num_candidates, H, W],   # SAM3 placement heatmap
    "rotation_6d": [B, 6],                   # 6D rotation representation
    "scale_relative": [B],                   # Relative scale (0.5x - 2.0x)
    "seg_hidden": [B, 4096],                # <SEG> token hidden state
}
```

**Output from `predict()`**:
```python
{
    "heatmap": [B, H, W],                    # 2D placement heatmap (best candidate)
    "binary_heatmap": [B, H, W],             # Binary mask after threshold
    "rotation_deg": [B],                     # -180 to 180 degrees
    "scale_relative": [B],                   # 0.5 to 2.0
    "qwen_response": str,                    # Generated Qwen response
    "best_candidate_idx": int,               # Selected candidate index
}
```

**Usage**:
```python
model = SAMQPlacementModel(
    sam_checkpoint_path="./outputs/sam3_checkpoint_final.pt",
    adapter_checkpoint_path="./outputs/adapter_checkpoint_final.pt",
    qwen_model_name="./models/qwen3_vl",
    qwen_lora_path="./outputs/lora_weights",
    sam3_input_dim=256,
    qwen_hidden_dim=4096,
    adapter_hidden_dim=512,
    action_head_config={"heatmap_size": 64},
)
model.load_all(eval_mode=True)

output = model.predict(
    plane_image=room_img,
    text_prompt="Place the chair near the table",
    images=[room_img, chair_img],
    threshold=0.5,
)
```

#### SEGActionHead

**Purpose**: Decode `<SEG>` hidden state into rotation and scale.

**Architecture**:
```
<SEG> hidden [B, 4096]
    ↓
Linear(4096→512) → LayerNorm → GELU
    ↓                    ↓
Heatmap Head        Rotation Head    Scale Head
    ↓                    ↓              ↓
Linear→[H*W]       Linear→Tanh    Linear→Sigmoid
    ↓              →[-1,1]×180   →[0,1]×1.5+0.5
Heatmap         [-180, 180]°     [0.5, 2.0]x
```

**Implementation**: `src/models/vla/unified_scale_vla.py`

---

## Data Flow

### Stage 1: Qwen3-VL LoRA Fine-Tuning

```
annotations.json
      |
ObjectPlacementDataset + DataLoader
      |
+----------------------------+
| Qwen3VLEncoder.forward()   |
|  1. Build messages with images|
|  2. Apply chat template    |
|  3. Forward through Qwen   |
|  4. Compute LM loss on <SEG>|
+------------|---------------+
             v
SFTTrainer (HuggingFace)
             |
+----------------------------+
| - AdamW optimizer          |
| - Cosine LR scheduler      |
| - Save LoRA to lora_weights/|
| - Auto-extract seg_features |
+----------------------------+
```

### Stage 2: Adapter + SAM3 Decoder Training

```
annotations.json + seg_features/
      |
ObjectPlacementDataset + DataLoader
      |
+----------------------------+
| For each sample in batch:  |
|  1. Read seg_hidden (skip Qwen) |
|  2. seg_hidden -> Adapter -> SAM3 prompts|
|  3. plane_image + prompts -> SAM3 decoder|
|  4. seg_hidden -> SEGActionHead         |
|  5. Output: heatmap + rotation + scale  |
+------------|---------------+
             v
PlacementLoss(heatmap, rotation, scale)
             |
+----------------------------+
| Loss Components:           |
|  - Dice Loss (weight=1.0)  |
|  - BCE Loss (weight=1.0)   |
|  - Rotation L1 (weight=0.5)|
|  - Scale L1 (weight=0.3)   |
+------------|---------------+
             v
backward() -> optimizer.step()
             |
Save: adapter_checkpoint_*.pt, sam3_checkpoint_*.pt
```

### Inference

```
plane_image + object_image + text
             |
SAMQPlacementModel.predict()
             |
+----------------------------+
| 1. Qwen3-VL -> <SEG> token|
| 2. <SEG> -> Adapter -> SAM3|
| 3. <SEG> -> SEGActionHead  |
| 4. Best candidate selection|
| 5. Qwen response generation|
+------------|---------------+
             v
Results: {heatmap, binary_heatmap, rotation_deg, scale_relative, qwen_response}
```

---

## File Structure

```
SAM-Q/
|-- configs/                     # Configuration files
|
|-- src/
|   +-- models/                  # Model architectures
|   |   +-- encoders/           # Encoder modules
|   |   +-- loaders/            # SAM3 modules
|   |   +-- adapters/           # Adapter modules
|   |   +-- vla/                # VLA action output
|   |   +-- placement_model.py  # Main model (unified predict)
|   |
|   +-- pretreatment/            # Data generation
|   |
|   +-- data/                    # Data pipeline
|   |
|   +-- train/                   # Training framework
|   |
|   +-- inference/               # Inference utilities
|   |
|   +-- utils/                   # Utilities
|   |
|   +-- sam3/                    # SAM3 model (submodule)
|
|-- scripts/                     # Helper scripts
|
|-- main.py                      # CLI entry point
|-- README.md                    # Documentation
|-- ARCHITECTURE.md              # This file
|-- CONTRIBUTING.md              # Contribution guide
+-- requirements.txt             # Dependencies
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
**Last Updated**: 2026-04-30
