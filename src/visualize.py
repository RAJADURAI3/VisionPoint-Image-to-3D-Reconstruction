import open3d as o3d
import numpy as np

def show_points(points, color=(0.2, 0.7, 1.0), screenshot_path=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pc)
    vis.poll_events()
    vis.update_renderer()

    if screenshot_path:
        vis.capture_screen_image(screenshot_path)
        print(f"✅ Screenshot saved to {screenshot_path}")

    vis.run()
    vis.destroy_window()

def compare_points(pred_points, gt_points,
                   pred_color=(0.2, 0.7, 1.0),
                   gt_color=(1.0, 0.5, 0.2),
                   offset=1.5,
                   screenshot_path=None):
    pred_pc = o3d.geometry.PointCloud()
    pred_pc.points = o3d.utility.Vector3dVector(pred_points)
    pred_pc.colors = o3d.utility.Vector3dVector(np.tile(pred_color, (pred_points.shape[0], 1)))

    gt_pc = o3d.geometry.PointCloud()
    gt_pc.points = o3d.utility.Vector3dVector(gt_points)
    gt_pc.colors = o3d.utility.Vector3dVector(np.tile(gt_color, (gt_points.shape[0], 1)))
    gt_pc.translate((offset, 0, 0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pred_pc)
    vis.add_geometry(gt_pc)
    vis.poll_events()
    vis.update_renderer()

    if screenshot_path:
        vis.capture_screen_image(screenshot_path)
        print(f"✅ Screenshot saved to {screenshot_path}")

    vis.run()
    vis.destroy_window()
