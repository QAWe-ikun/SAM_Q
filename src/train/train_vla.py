"""
VLA Training Script
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.sam2qhmvpl_system import SAM2QVLAIncremental
from models.placement_model import VLALoss  # Keep for compatibility
from data.vla_dataset import VLADataset
from torch.utils.data import DataLoader


class VLATrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = self._init_model()
        self.criterion = VLALoss(
            mask_weight=config.get("mask_weight", 1.0),
            text_weight=config.get("text_weight", 0.5),
        )
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def _init_model(self):
        model_config = self.config.get("model", {})
        
        # Use the new SAM²-Q-VLA-HMVP incremental system
        model = SAM2QVLAIncremental(
            sam_high_res=model_config.get("sam_high_res", 1024),
            sam_low_res=model_config.get("sam_low_res", 256),
            hmvp_max_level=model_config.get("hmvp_max_level", 4),
            hmvp_base_resolution=model_config.get("hmvp_base_resolution", 8),
            lifting_hidden_dim=model_config.get("lifting_hidden_dim", 256),
            num_candidates=model_config.get("num_candidates", 5),
            optimization_steps=model_config.get("optimization_steps", 10),
            incremental_updates=model_config.get("incremental_updates", True),
        )

        # For now, keep the same freezing strategy as before
        # In the future, we might want to freeze different parts
        for name, param in model.named_parameters():
            # Freeze certain components based on configuration
            if self.config.get("freeze_sam2_components", True):
                if "sam2_encoder" in name:
                    param.requires_grad = False
            if self.config.get("freeze_qwen_components", True):
                if "qwen3vl" in name:
                    param.requires_grad = False

        model.to(self.device)
        return model

    def _init_optimizer(self):
        opt_config = self.config.get("optimizer", {})
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(trainable_params, lr=opt_config.get("lr", 1e-4), weight_decay=opt_config.get("weight_decay", 1e-4))

    def _init_scheduler(self):
        sched_config = self.config.get("scheduler", {})
        return CosineAnnealingLR(self.optimizer, T_max=sched_config.get("T_max", 100), eta_min=sched_config.get("eta_min", 1e-6))

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {self.current_epoch}"):
            masks = batch["masks"].to(self.device)

            batch_loss = 0.0
            for i in range(len(batch["plane_images"])):
                # Adapt to new model interface
                # The new model expects object_img, scene_img, and text_query
                object_img = batch["object_images"][i].unsqueeze(0)  # Add batch dimension
                scene_img = batch["plane_images"][i].unsqueeze(0)    # Add batch dimension
                text_query = [batch["text_prompts"][i]]              # Make it a list
                
                output = self.model(
                    object_img=object_img,
                    scene_img=scene_img,
                    text_query=text_query,
                )
                
                # Extract the relevant outputs for loss calculation
                # This might need adjustment based on what the new model outputs
                predicted_poses = output["best_pose"]
                
                # For now, we'll use a simplified loss based on the original criterion
                # In practice, you'd want to define a new loss function for 3D placement
                loss_dict = self._calculate_placement_loss(output, masks[i:i+1])
                batch_loss += loss_dict["total"]

            batch_loss = batch_loss / len(batch["plane_images"])
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()

        return {"train_loss": total_loss / len(dataloader)}

    def _calculate_placement_loss(self, model_output, target_masks):
        """Calculate loss for the new 3D placement model"""
        # This is a simplified loss calculation
        # In practice, you'd want to implement a proper loss for 3D placement
        
        # For now, return a dummy loss dictionary to maintain compatibility
        batch_size = model_output["best_pose"].size(0)
        dummy_loss = torch.tensor(0.1, device=model_output["best_pose"].device, requires_grad=True)
        
        return {
            "total": dummy_loss,
            "pose_reg": dummy_loss * 0.5,
            "collision": dummy_loss * 0.5
        }

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}")
            self._save_checkpoint(epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_dataset = VLADataset(config["data_dir"], split="train")
    train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 4), shuffle=True, num_workers=config.get("num_workers", 4))

    trainer = VLATrainer(config)
    trainer.train(train_loader, config.get("num_epochs", 100))


if __name__ == "__main__":
    main()



