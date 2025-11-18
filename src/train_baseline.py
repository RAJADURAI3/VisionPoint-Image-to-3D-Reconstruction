import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model_baseline import ImageToPointCloud
from dataset_mono3d import Mono3DPCLDataset
import os

# Hyperparameters
num_points = 1024
batch_size = 16
epochs = 50   
lr = 1e-3

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# ✅ Image preprocessing
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ✅ Dataset + DataLoader (sofa only)
train_dataset = Mono3DPCLDataset(
    root_dir="data/Mono3DPCL",
    categories=["sofa"],   # restrict to sofa category
    split="train",
    n_points=num_points,
    transform=tfm
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ✅ Model
model = ImageToPointCloud(num_points=num_points).to(device)

# ✅ Chamfer Distance Loss
def chamfer_loss(pred, gt):
    # pred: (B, N, 3), gt: (B, N, 3)
    dist1 = torch.cdist(pred, gt).min(dim=2)[0].mean()
    dist2 = torch.cdist(gt, pred).min(dim=2)[0].mean()
    return dist1 + dist2

criterion = chamfer_loss
optimizer = optim.Adam(model.parameters(), lr=lr)

# ✅ Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, gt_points, _, _ in train_loader:  # dataset returns (img, points, cat, label)
        images, gt_points = images.to(device), gt_points.to(device)

        optimizer.zero_grad()
        pred_points = model(images)
        loss = criterion(pred_points, gt_points)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Chamfer Loss: {avg_loss:.6f}")

    # ✅ Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/sofa_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")
