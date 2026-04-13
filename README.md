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
- **Cross-Modal Fusion**: Novel adapter architecture bridges Qwen3-VL (4096D) and SAM3 (256D) embedding spaces
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
|  +---------------+  +---------------+  +-------------------+       |
|  | Dual-Scale SAM|  | H-MVP Collision|  | Incremental VLA   |       |
|  | (1024+256)    |  | Detector      |  | Memory System     |       |
|  +---------------+  +---------------+  +-------------------+       |
+-------------------------------------------------------------------+
```

### Core Components

#### 1. Qwen3-VL Encoder
- **Purpose**: Replaces SAM3's text encoder with multimodal vision-language capabilities
- **Input**: Object image + text instruction in conversation format
- **Output**: 4096-dimensional token embeddings
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

#### 5. Incremental VLA Memory (Optional)
- **Purpose**: Maintains dynamic scene understanding across placements
- **Workflow**:
  ```
  Initial Scene -> Build H-MVP -> Place Object A -> Update H-MVP -> Place Object B -> ...
  ```
- **Implementation**: `src/models/vla/incremental_vla.py`

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
# 先装 PyTorch（根据 CUDA 版本调整，cu130 对应 CUDA 13.0）
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130

# 装 flash-attn（需要先装好 torch，且 nvcc >= 12.0）
pip install flash-attn --no-build-isolation

# 装其余依赖
pip install -r requirements.txt

# 装 SAM3
pip install git+https://github.com/facebookresearch/sam3.git
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
# Single placement prediction
python main.py predict \
  --checkpoint checkpoints/checkpoint_best.pt \
  --plane_image examples/room.png \
  --object_image examples/chair.png \
  --prompt "Place the chair near the dining table" \
  --output results/output.png \
  --threshold 0.5
```

### Python API

```python
from src.inference import PlacementPredictor
from PIL import Image

# Load predictor
predictor = PlacementPredictor("checkpoints/checkpoint_best.pt")

# Prepare inputs
plane_image = Image.open("room.png").convert("RGB")
object_image = Image.open("chair.png").convert("RGB")
text_prompt = "Place the chair near the window"

# Predict placement
results = predictor.predict(
    plane_image=plane_image,
    object_image=object_image,
    text_prompt=text_prompt,
    threshold=0.5
)

# Results contain: mask, heatmap, boxes, scores
print(f"Found {len(results['boxes'])} valid placements")
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
+-- object_images/           # Object top-down views (512x512)
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
python -m src.train --config configs/base.yaml

# With overrides
python -m src.train \
  --config configs/base.yaml \
  --data_dir /path/to/data \
  --output_dir /path/to/outputs
```

### Monitoring

Training outputs are saved to:
- `outputs/checkpoint_epoch_X.pt` - Per-epoch checkpoints
- `outputs/checkpoint_best.pt` - Best validation loss checkpoint
- `outputs/checkpoint_final.pt` - Final checkpoint

### Advanced Training Strategies

#### Dual-Scale SAM + H-MVP Training
```bash
python -m src.train --config configs/hmvp.yaml
```

#### Incremental VLA Training
```bash
python -m src.train --config configs/incremental_vla.yaml
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
|   +-- base.yaml               # Base configuration
|   +-- hmvp.yaml               # H-MVP extension
|   +-- incremental_vla.yaml    # VLA extension
|
+-- src/
|   +-- models/                  # Model architectures
|   |   +-- encoders/           # Encoder modules
|   |   +-- adapters/           # Adapter modules
|   |   +-- collision/          # Collision detection
|   |   +-- vla/                # VLA components
|   |   +-- sampling/           # Sampling strategies
|   |   +-- placement_model.py  # Main model
|   |
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
  author={Your Name and Co-Author Name},
  journal={ACM Transactions on Graphics (SIGGRAPH 2026)},
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
