import os
import torch
import numpy as np
import open3d as o3d
import csv
import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_mono3d import Mono3DPCLDataset
from model_baseline import ImageToPointCloud
import argparse
from visualize import compare_points   # <-- added import

# ✅ Chamfer distance
def chamfer_loss(pred, gt):
    dist1 = torch.cdist(pred, gt).min(dim=2)[0].mean()
    dist2 = torch.cdist(gt, pred).min(dim=2)[0].mean()
    return dist1 + dist2

def save_pointcloud(points, save_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(save_path, pcd)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file. If not provided, latest in 'checkpoints/' will be used.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to search for checkpoints if none is specified.")
    parser.add_argument("--csv", type=str, default="eval_results.csv",
                        help="CSV file to log evaluation results")
    args = parser.parse_args()

    root = r"C:/Users/Hi/OneDrive/Desktop/Multi 3D-Reconstruction/data/Mono3DPCL"

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    ds = Mono3DPCLDataset(root_dir=root, categories=None, split="test", transform=tfm)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Evaluating on:", device)

    # ✅ Load chosen checkpoint
    if args.checkpoint is None:
        checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {args.checkpoint_dir}")
        checkpoints.sort(key=os.path.getmtime)
        args.checkpoint = checkpoints[-1]
        print(f"Using latest checkpoint: {args.checkpoint}")
    elif not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint} not found.")

    model = ImageToPointCloud(num_points=1024).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    out_dir = "predictions"
    os.makedirs(out_dir, exist_ok=True)

    category_losses = {}
    overall_losses = []

    with torch.no_grad():
        for i, (img, pcl, cat, label) in enumerate(dl):
            img, pcl = img.to(device), pcl.to(device)
            pred = model(img)
            loss = chamfer_loss(pred, pcl).item()
            overall_losses.append(loss)

            cat = cat[0]
            category_losses.setdefault(cat, []).append(loss)

            pred_np = pred.squeeze(0).cpu().numpy()
            save_path = os.path.join(out_dir, f"{cat}_pred_{i}.ply")
            save_pointcloud(pred_np, save_path)

            # ✅ Instant visualization + screenshot for first 3 samples
            if i < 3:
                gt_np = pcl.squeeze(0).cpu().numpy()
                screenshot_file = os.path.join(out_dir, f"compare_{cat}_{i}.png")
                compare_points(pred_np, gt_np, screenshot_path=screenshot_file)

    print("\n=== Evaluation Results ===")
    results = []
    for cat, losses in category_losses.items():
        avg_loss = np.mean(losses)
        print(f"{cat}: Chamfer distance = {avg_loss:.6f}")
        results.append([cat, avg_loss])
    overall_avg = np.mean(overall_losses)
    print(f"Overall average Chamfer distance = {overall_avg:.6f}")
    results.append(["Overall", overall_avg])

    # ✅ Log results to CSV
    write_header = not os.path.exists(args.csv)
    with open(args.csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Category", "ChamferDistance", "Checkpoint"])
        for cat, avg_loss in results:
            writer.writerow([cat, avg_loss, args.checkpoint])

    print(f"\nResults logged to {args.csv}")

if __name__ == "__main__":
    main()
