# SAM-Q: Segment Anything meets Vision-Language Models for Intelligent Object Placement

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#citation">Citation</a>
</p>

---
 
## Overview

**SAM-Q** is a novel framework that unifies **Segment Anything Model 3 (SAM3)** and **Qwen3-VL** vision-language models for **semantically-aware object placement** in indoor scenes. Given a top-down view of a room, an object image, and a natural language instruction, SAM-Q predicts optimal placement positions that respect both geometric constraints and semantic affordances.

### Key Features

- **Language-Guided Placement**: Natural language instructions control placement semantics
- **<SEG> Token Parallel Output**: Single `<SEG>` token feeds two parallel branches — SAM3 Decoder (placement heatmap) + SEGActionHead (rotation + scale)
- **Cross-Modal Fusion**: Novel adapter architecture bridges Qwen3-VL (4096D) and SAM3 (256D) embedding spaces
- **LoRA/QLoRA Fine-Tuning**: Parameter-efficient tuning with multi-SEG token support; trainable <0.1% of Qwen3-VL parameters
- **Parameter-Efficient**: Freezes foundation models, trains only <5% parameters

### Method Comparison

| Method | Language Understanding | Incremental | Real-time |
|--------|----------------------|-------------|-----------|
| Prior work [Chen et al. 2024] | No | No | Yes |
| VLA-Placement [Wang et al. 2025] | Yes | No | No |
| **SAM-Q (Ours)** | **Yes** | **Yes** | **Yes** |

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
```

### Core Components

#### 1. Qwen3-VL Encoder
- **Purpose**: Replaces SAM3's text encoder with multimodal vision-language capabilities
- **Input**: Object image + text instruction in conversation format
- **Output**: 4096-dimensional token embeddings
- **<SEG> Token Support**:
  - Single `<SEG>` mode: one token for single placement prediction
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

#### 4. VLA Action Output (Parallel with SAM3)
- **Purpose**: Outputs position (heatmap), rotation, and scale for intelligent placement
- **Architecture**: `<SEG>` token feeds two parallel branches:
  ```
  Qwen3-VL → <SEG> token
      ├→ SAM3 Decoder → placement heatmap (position)
      └→ SEGActionHead → rotation_deg + scale_relative
  ```
- **Key Design**:
  - Single `<SEG>` token serves as the shared representation
  - SAM3 generates the placement heatmap (2D position)
  - `SEGActionHead` generates rotation angle and relative scale
  - Unified `predict()` returns all outputs in one call
- **Implementation**:
  - `src/models/placement_model.py` (unified forward)

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


## Training

### Two-Stage Training Pipeline

SAM-Q uses a **two-stage training** strategy:

- **Stage 1**: Fine-tune Qwen3-VL with LoRA to generate `<SEG>` tokens
- **Stage 2**: Train Adapter + SAM3 Decoder for placement prediction (heatmap + rotation + scale)

### Prepare Dataset

#### Option 1: Use Existing Dataset

📖 **See**: [docs/DATASET.md](docs/DATASET.md) for details

#### Option 2: Auto-Generate Dataset from 3D Scenes

If you have SSR3D-FRONT dataset:

```bash
python main.py pretreat --config configs/pretreatment.yaml
```

📖 **See**: [docs/DATA_GENERATION.md](docs/DATA_GENERATION.md) for details.

**annotations.json format**:
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

### Run Training

```bash
# Stage 1: Qwen3-VL LoRA fine-tuning
python main.py train --config configs/stage1_qwen_lora.yaml

# Stage 2: Adapter + SAM3 Decoder training
python main.py train --config configs/stage2_decoder.yaml
```

### Training Outputs

After Stage 1:
- `outputs/lora_weights/` — Qwen3-VL LoRA adapter
- `data/seg_features/` — Auto-extracted `<SEG>` hidden states

After Stage 2:
- `outputs/adapter_checkpoint_best.pt` — Best adapter weights
- `outputs/adapter_checkpoint_final.pt` — Final adapter weights
- `outputs/sam3_checkpoint_best.pt` — Best SAM3 decoder weights (in original SAM3 format)
- `outputs/sam3_checkpoint_final.pt` — Final SAM3 decoder weights

---

## Fine-Tuning Qwen3-VL with <SEG> Tokens

SAM-Q supports parameter-efficient fine-tuning of Qwen3-VL using **LoRA** combined with **SA2VA-style <SEG> token bridging**.

### How <SEG> Tokens Work

The `<SEG>` token is a special token added to Qwen3-VL's vocabulary. During forward pass:

```
Input: [Image Tokens] "Place the chair near the table" <SEG>
       ↓
Self-Attention (causal mask): <SEG> attends to all preceding tokens
       ↓
<SEG> Hidden State: dynamically computed from image + text context (NOT a fixed vector)
       ↓
SEG Projector → SAM3 Decoder → Placement Mask
```

**Key insight**: The <SEG> token's hidden state varies based on input context — it learns *when* and *where* to trigger segmentation through joint training.

### Stage 1 Training with SFTTrainer

Stage 1 uses HuggingFace's `SFTTrainer` for efficient fine-tuning:

```bash
python main.py train --config configs/stage1_qwen_lora.yaml
```

After Stage 1 completes:
- LoRA weights saved to `outputs/lora_weights/`
- `<SEG>` hidden states auto-extracted to `data/seg_features/`

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

### Stage 2 Training with Seg Features

Stage 2 uses pre-extracted `<SEG>` features, skipping Qwen3-VL (~16GB VRAM saved):

```bash
python main.py train --config configs/stage2_decoder.yaml
```

Config for Stage 2:
```yaml
data:
  seg_feature_dir: "data/seg_features/"   # Pre-extracted features
loss:
  type: "placement"     # Dice + BCE + rotation + scale
```

---

## Project Structure

```
SAM-Q/
+-- configs/                     # Configuration files
|
+-- src/
|   +-- models/                  # Model architectures
|   |   +-- encoders/           # Qwen3-VL encoder
|   |   +-- loaders/           # SAM3 loader
|   |   +-- adapters/           # Cross-modal adapters
|   |   +-- vla/                # VLA action output
|   |   +-- placement_model.py  # Main model
|   |
|   +-- sam3/                    # SAM3 model
|   +-- data/                    # Data pipeline
|   +-- train/                   # Training framework
|   +-- pretreatment/            # Generating training data
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
| Semantic Alignment | 0.54 | **0.81** | +50.0% |
| Inference Time (s) | 0.15 | **0.12** | -20.0% |

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
- **Email**: 13610252512@139.com
- **Project Page**: [Coming Soon]

---

<p align="center">
  <strong>Star this repo if you find it helpful!</strong>
</p>
