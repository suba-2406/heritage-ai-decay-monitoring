#!/usr/bin/env python3
"""
Heritage AI Decay Monitoring - PyTorch Dataset Loader for SSD
Loads processed 320x320 images and VOC annotations.
Correctly handles Normal images as negative samples with 0 defect bounding boxes.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

CLASS_MAP = {
    "Normal": 0,
    "Crack": 1,
    "Moss": 2,
    "Seepage": 3
}

class HeritageMonumentDataset(Dataset):
    def __init__(self, processed_dir, split="train", transforms=None):
        self.processed_dir = processed_dir
        self.img_dir = os.path.join(processed_dir, "images")
        self.split = split
        self.transforms = transforms

        manifest_path = os.path.join(processed_dir, "dataset_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            all_items = json.load(f)

        self.items = [it for it in all_items if it["split"] == split]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = os.path.join(self.img_dir, item["processed_image"])
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for b in item["boxes"]:
            cls_name = b["name"]
            if cls_name in CLASS_MAP and cls_name != "Normal":
                boxes.append(b["bbox"])
                labels.append(CLASS_MAP[cls_name])

        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            # Negative sample (Normal image) -> empty boxes and labels
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            labels_tensor = torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([item["id"]]),
            "filename": item["filename"],
            "dominant_class": item["dominant_class"],
            "is_normal": item["is_normal"]
        }

        # Convert image to Tensor (range [0, 1])
        img_tensor = TF.to_tensor(image)

        # Optional augmentations
        if self.transforms is not None:
            img_tensor, target = self.transforms(img_tensor, target)

        return img_tensor, target

def collate_fn(batch):
    """Custom collate function for object detection batches with varying box counts."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
