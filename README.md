# SAM-Q: Segment Anything meets Vision-Language Models for Intelligent Object Placement

<p align="center">
  <img src="assets/teaser.png" alt="SAM-Q Teaser" width="800"/>
</p>

<p align="center">
  <strong>SIGGRAPH 2026 (Under Review)</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#citation">Citation</a>
</p>

---
 
## Overview

**SAM-Q** is a novel framework that unifies **Segment Anything Model 3 (SAM3)** and **Qwen3-VL** vision-language models for **semantically-aware object placement** in indoor scenes. Given a top-down view of a room, an object image, and a natural language instruction, SAM-Q predicts optimal placement positions that respect both geometric constraints and semantic affordances.

### Key Features

- **Language-Guided Placement**: Natural language instructions control placement semantics
- **[SEG] Token Parallel Output**: Single `[SEG]` token feeds two parallel branches — SAM3 Decoder (placement heatmap) + SEGActionHead (rotation + scale)
- **Cross-Modal Fusion**: Novel adapter architecture bridges Qwen3-VL (4096D) and SAM3 (256D) embedding spaces
- **LoRA/QLoRA Fine-Tuning**: Parameter-efficient tuning with multi-SEG token support; trainable <0.1% of Qwen3-VL parameters
- **Hierarchical Collision Detection**: H-MVP (Hierarchical Multi-View Projection) enables 3D-aware placement
- **Incremental Memory**: Dynamic scene understanding that updates with each placement
- **Parameter-Efficient**: Freezes foundation models, trains only <5% parameters

### Method Comparison

| Method | Language Understanding | 3D Collision | Incremental | Real-time |
|--------|----------------------|--------------|-------------|-----------|
| Prior work [Chen et al. 2024] | No | Yes | No | Yes |
| VLA-Placement [Wang et al. 2025] | Yes | No | No | No |
| **SAM-Q (Ours)** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## Architecture

### System Overview

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
|  |  Plane & Text Embeddings -> SAM3 Detector -> Placement Masks|    |
|  +------------------------------------------------------------+    |
+-------------------------------------------------------------------+
              |
              v
+-------------------------------------------------------------------+
|                   ADVANCED MODULES (Optional)                      |
|                                                                    |
|  +---------------+  +-------------------+  +-------------------+   |
|  | SEGActionHead |  | H-MVP Collision   |  | Incremental VLA   |   |
|  | (rotation +   |  | Detector          |  | Memory System     |   |
|  |  scale)       |  | (3D collision)    |  | (dynamic scene)   |   |
|  +---------------+  +-------------------+  +-------------------+   |
+-------------------------------------------------------------------+
```

### Core Components

#### 1. Qwen3-VL Encoder
- **Purpose**: Replaces SAM3's text encoder with multimodal vision-language capabilities
- **Input**: Object image + text instruction in conversation format
- **Output**: 4096-dimensional token embeddings
- **[SEG] Token Support**:
  - Single `[SEG]` mode: one token for single placement prediction
  - Multi `[SEG0]`~`[SEG63]` mode: SA2VA-style for multiple placements or complex spatial reasoning
  - Hidden states are **not fixed vectors** — they are dynamically computed via self-attention over the full image + text context
- **LoRA/QLoRA Fine-Tuning**:
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) + `gate_proj`, `up_proj`, `down_proj` (MLP)
  - Trainable parameters: <0.1% of 8B (~8M params)
  - VRAM: ~10GB (QLoRA 4-bit) / ~21GB (FP16 + Flash Attention 2) on RTX 4090
- **Implementation**: `src/models/encoders/qwen3vl_encoder.py`

#### 2. Cross-Modal Adapter
- **Purpose**: Projects Qwen3-VL embeddings to SAM3's embedding space
- **Architecture**:
  ```
  Input (4096D) -> Linear -> LayerNorm -> Cross-Attention (64 queries) -> Output Proj (256D)
  ```
- **Implementation**: `src/models/adapters/cross_modal_adapter.py`

#### 3. SAM3 Detector
- **Purpose**: Generates placement probability masks
- **Input**: Plane image embeddings + text embeddings
- **Output**: Binary placement masks with confidence scores
- **Implementation**: External (facebookresearch/sam3)

#### 4. H-MVP Collision Detector (Optional)
- **Purpose**: Hierarchical multi-view projection for 3D collision detection
- **Features**:
  - Depth pyramid construction (4 levels)
  - 6 orthogonal views per level
  - Early-exit optimization for fast inference
- **Implementation**: `src/models/collision/hmvp_collision_detector.py`

#### 5. VLA Action Output (Parallel with SAM3)
- **Purpose**: Outputs position (heatmap), rotation, and scale for intelligent placement
- **Architecture**: `[SEG]` token feeds two parallel branches:
  ```
  Qwen3-VL → [SEG] token
      ├→ SAM3 Decoder → placement heatmap (position)
      └→ SEGActionHead → rotation_deg + scale_relative
  ```
- **Key Design**:
  - Single `[SEG]` token serves as the shared representation
  - SAM3 generates the placement heatmap (2D position)
  - `SEGActionHead` generates rotation angle and relative scale
  - Unified `predict()` returns all outputs in one call
- **Implementation**:
  - `src/models/placement_model.py` (unified forward)
  - `src/models/vla/unified_scale_vla.py` (SEGActionHead)

---

## Installation

### Requirements

- **OS**: Linux (Ubuntu 20.04+, WSL2 支持) 或 Windows 10/11
- **Python**: 3.10+
- **GPU**: NVIDIA GPU，16GB+ 显存（推荐 A100/V100/RTX 4090）
- **CUDA Toolkit**: 12.0+（flash-attn 编译需要，`nvcc --version` 检查）

### Step 1: 创建环境

```bash
# 使用 conda（推荐）
conda create -n samq python=3.12
conda activate samq

# 或使用 venv
python -m venv samq-env
source samq-env/bin/activate  # Linux/Mac
# or
samq-env\Scripts\activate  # Windows
```

### Step 2: 安装依赖

```bash
# 先装 PyTorch（根据 CUDA 版本调整，cu124 对应 CUDA 12.4）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# 装 flash-attn（需要先装好 torch，且 nvcc >= 12.0）
# 如果仍不行，在线下载WHL，离线安装
pip install flash-attn --no-build-isolation

# 装其余依赖
pip install -r requirements.txt
```

### Step 3: Download Models

```bash
# Using ModelScope (recommended for China users)
python scripts/download_models.py

# Or using HuggingFace
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/qwen3_vl
huggingface-cli download facebook/sam3 --local-dir models/sam3
```

### Step 4: Verify Installation

```bash
# Run sanity checks
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Quick Start

### Basic Inference

```bash
# Single placement prediction (uses defaults)
python main.py predict \
  --checkpoint checkpoints/checkpoint_best.pt \
  --plane_image examples/room.png \
  --object_image examples/chair.png \
  --prompt "Place the chair near the dining table" \
  --output results/ \
  --threshold 0.5

# With config file
python main.py predict \
  --checkpoint checkpoints/checkpoint_best.pt \
  --config configs/config.yaml \
  --plane_image examples/room.png \
  --object_image examples/chair.png \
  --prompt "Place the chair near the dining table"
```

**Note:** `--object_image` is used internally as part of the `images` list alongside the plane image. The model receives `[plane_image, object_image]` for multi-image reasoning.

### Python API

```python
from src.models import SAMQPlacementModel
from PIL import Image

# Load model
model = SAMQPlacementModel(
    qwen_model_name="./models/qwen3_vl",
    sam3_input_dim=256,
    qwen_hidden_dim=4096,
    adapter_hidden_dim=512,
    device="cuda",
    action_head_config={"heatmap_size": 64},
)

# Load checkpoint
checkpoint = torch.load("checkpoints/checkpoint_best.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Predict
room_img = Image.open("examples/room.png").convert("RGB")
chair_img = Image.open("examples/chair.png").convert("RGB")

output = model.predict(
    plane_image=room_img,
    text_prompt="Place the chair near the dining table",
    images=[room_img, chair_img],
    threshold=0.5,
)

# Results
print(f"Scale: {output['scale_relative']}")
```

---


## Training

### Prepare Dataset

```
data/
+-- annotations.json          # Metadata with splits
+-- plane_images/            # Room top-down views (1024x1024)
|   +-- scene_001.png
|   +-- ...
+-- object_images/           # Object top-down views (1024x1024)
|   +-- obj_001.png
|   +-- ...
+-- masks/                   # Ground truth placement masks
    +-- scene_001_mask.png
    +-- ...
```

**annotations.json format**:
```json
[
  {
    "scene_id": "scene_001",
    "object_id": "obj_001",
    "plane_image_path": "plane_images/scene_001.png",
    "object_image_path": "object_images/obj_001.png",
    "mask_path": "masks/scene_001_mask.png",
    "text_prompt": "Place the chair near the table",
    "split": "train"
  }
]
```

### Run Training

```bash
# Basic training
python main.py train --config configs/config.yaml

# With overrides
python main.py train \
  --config configs/config.yaml \
  --data_dir /path/to/data \
  --output_dir /path/to/outputs
```

### Monitoring

Training outputs are saved to:
- `outputs/checkpoint_epoch_X.pt` - Per-epoch checkpoints
- `outputs/checkpoint_best.pt` - Best validation loss checkpoint
- `outputs/checkpoint_final.pt` - Final checkpoint

### Training with H-MVP or Incremental Memory

To enable H-MVP collision detection, edit `configs/config.yaml`:
```yaml
advanced:
  hmvp:
    enabled: true
```

To enable incremental H-MVP memory:
```yaml
advanced:
  incremental_hmvp:
    enabled: true
```

---

## Fine-Tuning Qwen3-VL with [SEG] Tokens

SAM-Q supports parameter-efficient fine-tuning of Qwen3-VL using **LoRA** combined with **SA2VA-style [SEG] token bridging**.

📖 **完整指南**: [docs/FINETUNE_QWEN3VL.md](docs/FINETUNE_QWEN3VL.md)

### How [SEG] Tokens Work

The `[SEG]` token is a special token added to Qwen3-VL's vocabulary. During forward pass:

```
Input: [Image Tokens] "Place the chair near the table" [SEG]
       ↓
Self-Attention (causal mask): [SEG] attends to all preceding tokens
       ↓
[SEG] Hidden State: dynamically computed from image + text context (NOT a fixed vector)
       ↓
SEG Projector → SAM3 Decoder → Placement Mask
```

**Key insight**: The [SEG] token's hidden state varies based on input context — it learns *when* and *where* to trigger segmentation through joint training.

### Multi-SEG Mode

For complex scenarios (multiple placements, spatial reasoning):

```python
# Single SEG (most common)
seg_hidden, _ = encoder.generate_with_seg(
    object_image=obj_img,
    text_prompt="放一把椅子",
    force_only=True,
    num_seg=1,  # output: [B, 4096]
)

# Multi SEG (advanced, SA2VA-style)
seg_hidden, _ = encoder.generate_with_seg(
    object_image=obj_img,
    text_prompt="放一把椅子和一张桌子",
    force_only=True,
    num_seg=8,  # output: [B, 8, 4096]
)
```

### LoRA Configuration

```python
from src.models.encoders.qwen3vl_encoder import Qwen3VLEncoder

encoder = Qwen3VLEncoder(model_name="./models/qwen3_vl")
encoder.load_model()
encoder.enable_finetuning(
    lora_r=64,           # LoRA rank
    lora_alpha=128,      # scaling factor (typically 2x rank)
    lora_dropout=0.05,   # regularization
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",       # MLP layers
    ],
    use_qlora=False,     # Set True for 4-bit quantization
)
# Output: Trainable params: 8,388,608 / 8,000,000,000 (0.10%)
```

### VRAM Requirements (RTX 4090)

| Configuration | VRAM | Status |
|--------------|------|--------|
| FP16 + Flash Attention 2 | ~21 GB | ✅ Inference only, batch=1 |
| QLoRA 4-bit + Flash Attention 2 | ~10 GB | ✅ Training, batch=1 |
| QLoRA 4-bit + gradient ckpt | ~8 GB | ✅ Training, comfortable |

### Training Workflow

```python
# 1. Prepare training batch
batch = encoder.prepare_training_batch(
    object_image=obj_img,
    text_prompt="在餐桌旁放椅子",
    num_seg=8,
)
# batch = {'input_ids': ..., 'attention_mask': ..., 'labels': ...}

# 2. Forward pass (training mode returns logits + loss)
output = encoder(
    object_image=obj_img,
    text_prompt=text,
    labels=batch["labels"],
)

# 3. Backward + optimizer step (only LoRA params updated)
loss = output["loss"]
loss.backward()
optimizer.step()

# 4. Save LoRA adapter
encoder.save_finetuned_model("./checkpoints/qwen3_vl_seg_lora")
```

---

## Inference

### Batch Inference

```bash
# Process multiple samples
python main.py predict \
  --checkpoint checkpoints/checkpoint_best.pt \
  --input data/test_samples/ \
  --output results/batch_results/
```

### Output Format

Inference results are saved as:
- **Visualization**: PNG with overlayed heatmap and bounding boxes
- **JSON metadata**:
  ```json
  {
    "scene_id": "scene_001",
    "num_placements": 3,
    "placements": [
      {
        "box": [x1, y1, x2, y2],
        "score": 0.95
      }
    ]
  }
  ```

---

## Project Structure

```
SAM-Q/
+-- configs/                     # Configuration files
|   +-- config.yaml               # Configuration
|   +-- stage1_qwen_lora.yaml               # Configuration for train stage 1
|   +-- stage2_decoder.yaml               # Configuration for train stage 2
|
+-- src/
|   +-- models/                  # Model architectures
|   |   +-- encoders/           # Qwen3-VL encoder
|   |   +-- adapters/           # Cross-modal adapters
|   |   +-- collision/          # H-MVP + incremental memory
|   |   +-- vla/                # VLA action output
|   |   +-- sampling/           # Sampling strategies
|   |   +-- placement_model.py  # Main model
|   |
|   +-- sam3/                    # SAM3 model
|   +-- data/                    # Data pipeline
|   +-- train/                   # Training framework
|   +-- inference/               # Inference utilities
|   +-- utils/                   # Utilities
|
+-- tests/                       # Unit tests
+-- scripts/                     # Helper scripts
+-- models/                      # Downloaded models
|   +-- qwen3_vl/
|   +-- sam3/
|
+-- main.py                      # CLI entry point
+-- README.md                    # Documentation
+-- ARCHITECTURE.md              # Architecture details
+-- CONTRIBUTING.md              # Contribution guide
+-- requirements.txt             # Dependencies
```

---

## Results

### Quantitative Evaluation

| Metric | Baseline | SAM-Q (Ours) | Improvement |
|--------|----------|--------------|-------------|
| IoU | 0.62 | **0.78** | +25.8% |
| Collision Rate | 18.5% | **6.2%** | -66.5% |
| Semantic Alignment | 0.54 | **0.81** | +50.0% |
| Inference Time (s) | 0.15 | **0.12** | -20.0% |

---

## Advanced Usage

### Custom Adapter Design

```python
from src.models.adapters import CrossModalAdapter

# Create custom adapter
adapter = CrossModalAdapter(
    qwen_dim=4096,
    sam3_dim=256,
    num_queries=64,
    hidden_dim=512,
)
```

### H-MVP Collision Detection

```python
from src.models.collision import HMVPCollisionDetector

detector = HMVPCollisionDetector(
    max_level=4,
    base_resolution=8,
    early_out_threshold=0.1
)

collision_score = detector.check_collision(
    scene_depths=scene_hmvp,
    object_depths=obj_hmvp,
    pose=predicted_pose
)
```

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

We follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Please ensure:
- All functions have type hints and docstrings
- Code is formatted with `black`
- Run `flake8` before committing

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{samq2026,
  title={SAM-Q: Segment Anything meets Vision-Language Models for Intelligent Object Placement},
  author={Weijia Li},
  journal={},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) - Segment Anything Model
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) - Vision-Language Model
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## Contact

- **Questions**: Open an issue on GitHub
- **Email**: your.email@example.com
- **Project Page**: [Coming Soon]

---

<p align="center">
  <strong>Star this repo if you find it helpful!</strong>
</p>
