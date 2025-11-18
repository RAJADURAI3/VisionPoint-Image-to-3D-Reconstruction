🚀 Overview
VisionPoint is a deep learning pipeline that reconstructs 3D point clouds directly from single RGB images. It demonstrates how 2D image features can be mapped into 3D geometry using a ResNet backbone and a regression head, evaluated with Chamfer Distance, and visualized interactively with Open3D.

This project highlights expertise in:

-Computer Vision & 3D Vision

-PyTorch & Torchvision

-Open3D Visualization

-Reproducible Training & Evaluation Pipelines

🏗️ Architecture

Backbone: ResNet (18/34/50) for image feature extraction.

Head: Fully connected regression layers mapping features → 3D coordinates.

Output: Point cloud of shape
```bash
(batch_size, num_points, 3)
```
Loss Function: Chamfer Distance — measures similarity between predicted and ground truth point clouds.

Evaluation: Automated benchmarking across checkpoints, CSV logging, and visualization.

📂 Project Structure

1.model_baseline.py

Defines the ImageToPointCloud model:

-ResNet backbone (configurable).

-Regression head outputs 3D point cloud coordinates.

-Flexible option for pretrained weights.

2.dataset_mono3d.py

Custom PyTorch dataset loader:

-Loads images + ground truth point clouds from Mono3DPCL dataset.

-Applies transforms (resize, normalization).

-Returns (image, point_cloud, category, label) tuples.

3.train_baseline.py

Training script:

-Loads the Mono3DPCL dataset.

-Trains the ImageToPointCloud model using Chamfer Distance loss.

-Saves checkpoints (.pth files) into checkpoints/.

-Supports resuming training from previous checkpoints.

4.eval_baseline.py

Evaluation script:

-Loads latest checkpoint (or all in folder).

-Runs inference on test set.

-Computes Chamfer Distance per category and overall.

-Saves predictions as .ply.

-Logs results into eval_results.csv.

-Visualizes predictions vs ground truth (with screenshots for first  few samples).

5.visualize.py

Visualization utilities using Open3D:

-show_points: Display a single point cloud.

-save_points: Save point cloud to .ply.

-compare_points: Compare predicted vs ground truth side by side.

-Supports automatic screenshot capture.

🏗️ Workflow

Training

```bash
python train_baseline.py --epochs 50 --batch_size 16 --checkpoint_dir checkpoints
```
-Trains the model on Mono3DPCL dataset.

-Saves checkpoints into checkpoints/.

Evaluation

```bash
python eval_baseline.py
```
-Loads latest checkpoint (or all in folder).

-Runs inference on test set.

-Saves .ply predictions in predictions/.

-Logs results in eval_results.csv.

-Generates screenshots for first few samples.

Visualization

-Inspect .ply files in MeshLab, Blender, or Open3D.

-Use visualize.py for interactive comparisons.

📈 Outputs

-Predictions: .ply files in predictions/

-Screenshots: .png comparison images (pred vs ground truth)

-Metrics: eval_results.csv with Chamfer Distance logs

⚙️ Dependencies

Python 3.8+

PyTorch

Torchvision

Open3D

NumPy

Install with:

```bash

pip install torch torchvision open3d numpy

```
🎯 Summary

Demonstrates a complete 3D reconstruction pipeline:

-Training on Mono3DPCL dataset.

-Evaluation with Chamfer Distance.

-Visualization with Open3D.

-Automated logging and reproducibility.


