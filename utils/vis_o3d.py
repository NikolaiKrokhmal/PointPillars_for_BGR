import numpy as np
import open3d as o3d
from .process_pollo import bbox3d2corners

COLORS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]

LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [2, 6], [7, 3], [1, 5], [4, 0]
]

def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    if npy.shape[1] > 3:
        density = npy[:, 3]
        colors = [[item, item, item] for item in density]
        ply.colors = o3d.utility.Vector3dVector(colors)
    return ply

def bbox_obj(points, color=[1, 0, 0]):
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES),
    )
    line_set.colors = o3d.utility.Vector3dVector([color for _ in LINES])
    return line_set

def vis_pc(pc, bbox=None, bbox_real=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_cloud = npy2ply(pc)
    vis.add_geometry(point_cloud)

    if bbox is not None:
        bbox_corners = bbox3d2corners(bbox)  # Add a dummy angle
        for box in bbox_corners:
            bbox_lines = bbox_obj(box, color=[1, 0, 0])             #Red
            vis.add_geometry(bbox_lines)

    if bbox_real is not None:
        bbox_real_corners = bbox3d2corners(bbox_real)  # Add a dummy angle
        for box in bbox_real_corners:
            bbox_real_lines = bbox_obj(box, color=[0, 1, 0])        #Green
            vis.add_geometry(bbox_real_lines)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.asarray([0, 0, 0])
    # Set the viewpoint AFTER the window is created
    vis.get_view_control().set_front([0, 0, 1])
    vis.get_view_control().set_lookat([0, 0, 0])
    vis.get_view_control().set_up([0, 1, 0])
    vis.get_view_control().set_zoom(0.5)

    vis.run()
    # vis.destroy_window()
    return

# Usage
# pc = np.random.rand(1000, 3)  # Replace with your actual point cloud data
# bbox = np.array([0, 0, 0, 1, 1, 1])  # Replace with your actual bbox data
# bbox_real = np.array([0, 0, 0, 2, 2, 2])  # Replace with your actual bbox_real data
# vis_PC(pc, bbox, bbox_real)
