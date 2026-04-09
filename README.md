# SAM-Q: Segment Anything meets Vision-Language Models for Intelligent Object Placement

<p align="center">
  <img src="assets/teaser.png" alt="SAM-Q Teaser" width="800"/>
</p>

<p align="center">
  <strong>SIGGRAPH 2026 (Under Review)</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> �?  <a href="#-architecture">Architecture</a> �?  <a href="#-installation">Installation</a> �?  <a href="#-quick-start">Quick Start</a> �?  <a href="#-training">Training</a> �?  <a href="#-inference">Inference</a> �?  <a href="#-dataset">Dataset</a> �?  <a href="#-citation">Citation</a>
</p>

---

## 📖 Overview

**SAM-Q** is a novel framework that unifies **Segment Anything Model 3 (SAM3)** and **Qwen3-VL** vision-language models for **semantically-aware object placement** in indoor scenes. Given a top-down view of a room, an object image, and a natural language instruction, SAM-Q predicts optimal placement positions that respect both geometric constraints and semantic affordances.

### Key Features

- 🧠 **Language-Guided Placement**: Natural language instructions control placement semantics
- 🔀 **Cross-Modal Fusion**: Novel adapter architecture bridges Qwen3-VL (4096D) and SAM3 (256D) embedding spaces
- 🎯 **Hierarchical Collision Detection**: H-MVP (Hierarchical Multi-View Projection) enables 3D-aware placement
- 🔄 **Incremental Memory**: Dynamic scene understanding that updates with each placement
- �?**Parameter-Efficient**: Freezes foundation models, trains only <5% parameters

### Method Comparison

| Method | Language Understanding | 3D Collision | Incremental | Real-time |
|--------|----------------------|--------------|-------------|-----------|
| Prior work [Chen et al. 2024] | �?| �?| �?| �?|
| VLA-Placement [Wang et al. 2025] | �?| �?| �?| �?|
| **SAM-Q (Ours)** | �?| �?| �?| �?|

---

## 🏗�?Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────�?�?                          INPUT LAYER                                �?�? ┌──────────────────�? ┌──────────────────�? ┌──────────────────�? �?�? �? Room Top-Down    �? �? Object Top-Down �? �? Text Instruction �? �?�? �? View (1024²)    �? �? View (512²)     �? �? (Natural Lang.)  �? �?�? └────────┬─────────�? └────────┬─────────�? └────────┬─────────�? �?└───────────┼─────────────────────┼─────────────────────┼────────────�?            �?                    �?                    �?            �?                    �?                    �?┌─────────────────────────────────────────────────────────────────────�?�?                         ENCODER LAYER (Frozen)                      �?�? ┌─────────────────────────�?        ┌───────────────────────────�? �?�? �? SAM3 Image Encoder     �?        �? Qwen3-VL Vision-Language  �? �?�? �? (Spatial Features)     �?        �? Encoder (Multimodal)      �? �?�? �? Output: 256-dim/patch  �?        �? Output: 4096-dim/token    �? �?�? └────────────┬────────────�?        └──────────────┬────────────�? �?�?              �?                                    �?               �?�?              �?                           ┌────────▼──────────�?   �?�?              �?                           �? Adapter Module    �?   �?�?              �?                           �? 4096 �?256 dims  �?   �?�?              �?                           �? + Cross-Attn     �?   �?�?              �?                           └───────────────────�?   �?└───────────────┼─────────────────────────────┬───────────────────────�?                �?                            �?                �?                            �?┌─────────────────────────────────────────────────────────────────────�?�?                      FUSION & DECODER LAYER                         �?�? ┌───────────────────────────────────────────────────────────────�? �?�? �? Embedding Concatenation �?SAM3 Detector �?Placement Masks    �? �?�? └───────────────────────────────────────────────────────────────�? �?└─────────────────────────────────────────────────────────────────────�?                �?                �?┌─────────────────────────────────────────────────────────────────────�?�?                    ADVANCED MODULES (Optional)                      �?�? ┌──────────────────�? ┌──────────────────�? ┌──────────────────�? �?�? �? Dual-Scale SAM  �? �? H-MVP Collision �? �? Incremental VLA �? �?�? �? (1024+256)      �? �? Detector        �? �? Memory System   �? �?�? └──────────────────�? └──────────────────�? └──────────────────�? �?└─────────────────────────────────────────────────────────────────────�?```

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
  Input (4096D) �?Linear �?LayerNorm �?Cross-Attention (64 queries) �?Output Proj (256D)
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
  Initial Scene �?Build H-MVP �?Place Object A �?Update H-MVP �?Place Object B �?...
  ```
- **Implementation**: `src/models/vla/incremental_vla.py`

---

## 📦 Installation

### Requirements

- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (A100/V100/RTX 4090 recommended)
- **CUDA**: 11.8+

### Step 1: Create Environment

```bash
# Using conda (recommended)
conda create -n samq python=3.10
conda activate samq

# Or using venv
python -m venv samq-env
source samq-env/bin/activate  # Linux/Mac
# or
samq-env\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install -r requirements.txt

# Install SAM3
pip install git+https://github.com/facebookresearch/sam3.git
```

### Step 3: Verify Installation

```bash
# Run sanity checks
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3: OK')"
```

---

## 🚀 Quick Start

### Download Pre-trained Models

```bash
# Create model directory
mkdir -p checkpoints

# Download Qwen3-VL-8B-Instruct (requires HuggingFace authentication)
# Visit: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

# SAM3 will be downloaded automatically on first run
```

### Basic Inference

```bash
# Single placement prediction
python main.py legacy_inference \
  --checkpoint checkpoints/checkpoint_best.pt \
  --plane_image examples/room.png \
  --object_image examples/chair.png \
  --prompt "Place the chair near the dining table" \
  --output results/output.png \
  --threshold 0.5
```

### Interactive Demo

```bash
# Launch Gradio demo (requires additional `pip install gradio`)
python main.py demo --config configs/config.yaml
```

### Python API

```python
from src.models.placement_model import SAM3PlacementModel
from PIL import Image

# Load model
model = SAM3PlacementModel(
    qwen_model_name="Qwen/Qwen3-VL-8B-Instruct",
    checkpoint_path="checkpoints/checkpoint_best.pt"
)
model.eval()

# Prepare inputs
plane_image = Image.open("room.png").convert("RGB")
object_image = Image.open("chair.png").convert("RGB")
text_prompt = "Place the chair near the window"

# Predict placement
with torch.no_grad():
    results = model.predict(
        plane_image=plane_image,
        object_image=object_image,
        text_prompt=text_prompt,
        threshold=0.5
    )

# Results contain: masks, boxes, scores, heatmap
print(f"Found {len(results['boxes'])} valid placements")
```

---

## 🎓 Training

### Prepare Dataset

```
data/
├── annotations.json          # Metadata with splits
├── plane_images/            # Room top-down views (1024x1024)
�?  ├── scene_001.png
�?  └── ...
├── object_images/           # Object top-down views (512x512)
�?  ├── obj_001.png
�?  └── ...
└── masks/                   # Ground truth placement masks
    ├── scene_001_mask.png
    └── ...
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

### Training Configuration

Edit `configs/config.yaml`:
```yaml
# Data
data_dir: "data/"
batch_size: 4
num_workers: 4

# Model
model:
  qwen_model_name: "Qwen/Qwen3-VL-8B-Instruct"
  sam3_input_dim: 256
  qwen_hidden_dim: 4096
  adapter_hidden_dim: 512

# Freeze strategies
freeze_qwen: true
freeze_sam3_image_encoder: true

# Optimizer
optimizer:
  lr: 1.0e-4
  weight_decay: 1.0e-4

# Training
num_epochs: 100
output_dir: "outputs/"
```

### Run Training

```bash
# Basic training
python -m src.train.train --config configs/config.yaml

# With overrides
python -m src.train.train \
  --config configs/config.yaml \
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
python -m src.train.train --config configs/sam2qhmvpl_config.yaml
```

#### Incremental VLA Training
```bash
python -m src.train.train --config configs/sam2qvla_incremental_config.yaml
```

---

## 🔍 Inference

### Batch Inference

```bash
# Process multiple samples
python main.py inference \
  --checkpoint checkpoints/checkpoint_best.pt \
  --input data/test_samples/ \
  --output results/batch_results/ \
  --config configs/config.yaml
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
        "score": 0.95,
        "mask_path": "results/scene_001_mask_0.png"
      }
    ]
  }
  ```

---

## 🗂�?Project Structure

```
SAM-Q/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── main.py                             # CLI entry point
�?├── configs/                            # Configuration files
�?  ├── config.yaml                     # Base configuration
�?  ├── sam2qhmvpl_config.yaml          # Dual-scale + H-MVP config
�?  ├── sam2qvla_incremental_config.yaml# Incremental VLA config
�?  └── vla_config.yaml                 # VLA-specific config
�?├── src/
�?  ├── models/                         # Model architectures
�?  �?  ├── __init__.py
�?  �?  ├── placement_model.py          # Main SAM3PlacementModel
�?  �?  ├── losses.py                   # Loss functions
�?  �?  �?�?  �?  ├── encoders/                   # Encoder modules
�?  �?  �?  ├── __init__.py
�?  �?  �?  └── qwen3vl_encoder.py      # Qwen3-VL wrapper
�?  �?  �?�?  �?  ├── adapters/                   # Adapter modules
�?  �?  �?  ├── __init__.py
�?  �?  �?  ├── base_adapter.py         # Basic MLP adapter
�?  �?  �?  ├── cross_modal_adapter.py  # Cross-attention adapter
�?  �?  �?  └── presence_token_adapter.py
�?  �?  �?�?  �?  ├── collision/                  # Collision detection
�?  �?  �?  ├── __init__.py
�?  �?  �?  └── hmvp_collision_detector.py
�?  �?  �?�?  �?  ├── vla/                        # VLA components
�?  �?  �?  ├── __init__.py
�?  �?  �?  └── incremental_vla.py
�?  �?  �?�?  �?  └── sampling/                   # Sampling strategies
�?  �?      ├── __init__.py
�?  �?      └── heatmap_guided_placer.py
�?  �?�?  ├── data/                           # Data pipeline
�?  �?  ├── __init__.py
�?  �?  ├── dataset.py                  # Base dataset
�?  �?  ├── vla_dataset.py              # VLA-specific dataset
�?  �?  └── transforms.py               # Data augmentation
�?  �?�?  ├── train/                          # Training framework
�?  �?  ├── __init__.py
�?  �?  ├── trainer.py                  # Trainer class
�?  �?  ├── optimizer.py                # Optimizer utilities
�?  �?  └── metrics.py                  # Evaluation metrics
�?  �?�?  ├── inference/                      # Inference utilities
�?  �?  ├── __init__.py
�?  �?  ├── predictor.py                # PlacementPredictor
�?  �?  └── visualizer.py               # Result visualization
�?  �?�?  └── utils/                          # Utilities
�?      ├── __init__.py
�?      ├── config.py                   # Configuration parser
�?      ├── asset_database.py           # 3D asset management
�?      └── common.py                   # Common utilities
�?├── scripts/                            # Helper scripts
�?  ├── download_data.sh
�?  ├── preprocess_data.py
�?  └── evaluate.py
�?└── assets/                             # Images for documentation
    └── teaser.png
```

---

## 📊 Results

### Quantitative Evaluation

| Metric | Baseline | SAM-Q (Ours) | Improvement |
|--------|----------|--------------|-------------|
| IoU �?| 0.62 | **0.78** | +25.8% |
| Collision Rate �?| 18.5% | **6.2%** | -66.5% |
| Semantic Alignment �?| 0.54 | **0.81** | +50.0% |
| Inference Time (s) | 0.15 | **0.12** | -20.0% |

### Qualitative Results

<p align="center">
  <img src="assets/results.png" alt="Qualitative Results" width="800"/>
</p>

---

## 🔧 Advanced Usage

### Custom Adapter Design

```python
from src.models.adapters import CrossModalAdapter

# Create custom adapter
adapter = CrossModalAdapter(
    qwen_dim=4096,
    sam3_dim=256,
    num_queries=64,      # Number of output queries
    hidden_dim=512,      # Hidden layer dimension
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

# Check collision
collision_score = detector.check_collision(
    scene_depths=scene_hmvp,
    object_depths=obj_hmvp,
    pose=predicted_pose
)
```

### Incremental VLA Memory

```python
from src.models.vla import IncrementalHMVPMemory

memory = IncrementalHMVPMemory(
    update_threshold=0.3,
    max_objects=50
)

# Initialize from scene
memory.initialize_from_scene(scene_image)

# Update after placement
memory.update_with_new_object(
    new_pose=predicted_pose,
    new_shape=object_shape
)
```

---

## 🤝 Contributing

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

## 📝 Citation

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [SAM3](https://github.com/facebookresearch/sam3) - Segment Anything Model
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) - Vision-Language Model
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 📧 Contact

- **Questions**: Open an issue on GitHub
- **Email**: your.email@example.com
- **Project Page**: [Coming Soon]

---

<p align="center">
  <strong>�?Star this repo if you find it helpful!</strong>
</p>
