import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import open3d as o3d

class Mono3DPCLDataset(Dataset):
    def __init__(self, root_dir, categories=None, split="train", transform=None, n_points=1024):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.n_points = n_points

        # If no categories specified, use all folders inside root_dir
        if categories is None:
            categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        self.samples = []
        for cat in categories:
            base = os.path.join(root_dir, cat, split)
            if not os.path.exists(base):
                continue
            for f in os.listdir(base):
                if f.endswith(".jpg") or f.endswith(".png"):
                    img_path = os.path.join(base, f)
                    ply_path = img_path.replace(".jpg", ".ply").replace(".png", ".ply")
                    txt_path = img_path.replace(".jpg", ".txt").replace(".png", ".txt")

                    if os.path.exists(ply_path):
                        label = None
                        if os.path.exists(txt_path):
                            with open(txt_path, "r") as fp:
                                label = fp.read().strip()  # read metadata/label
                        self.samples.append((img_path, ply_path, txt_path, label, cat))

        print(f"Loaded {len(self.samples)} {split} samples from categories: {categories}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ply_path, txt_path, label, cat = self.samples[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # Load point cloud
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        # Downsample to fixed number of points
        if len(points) > self.n_points:
            choice = np.random.choice(len(points), self.n_points, replace=False)
            points = points[choice]
        elif len(points) < self.n_points:
            pad = np.zeros((self.n_points - len(points), 3), dtype=np.float32)
            points = np.vstack((points, pad))

        points = torch.from_numpy(points)

        # Return image, point cloud, category, and optional label
        return img, points, cat, label
