data_process/align.py
```python
import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import pickle
import trimesh
import cv2
import json
import torch
import os
from utils.align_util import (
    render_multi_images,
    render_image,
    as_mesh,
    project_2d_to_3d,
    plot_mesh_with_points,
    plot_image_with_points,
    select_point,
)
from match_pairs import image_pair_matching
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import KDTree

VIS = True
parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name
output_dir = f"{base_path}/{case_name}/shape/matching"


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def pose_selection_render_superglue(
    raw_img, fov, mesh_path, mesh, crop_img, output_dir
):
    # Calculate suitable rendering radius
    bounding_box = mesh.bounds
    max_dimension = np.linalg.norm(bounding_box[1] - bounding_box[0])
    radius = 2 * (max_dimension / 2) / np.tan(fov / 2)

    # Render multimle images and feature matching
    colors, depths, camera_poses, camera_intrinsics = render_multi_images(
        mesh_path,
        raw_img.shape[1],
        raw_img.shape[0],
        fov,
        radius=radius,
        num_samples=8,
        num_ups=4,
        device="cuda",
    )
    grays = [cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) for color in colors]
    # Use superglue to match the features
    best_idx, match_result = image_pair_matching(
        grays, crop_img, output_dir, viz_best=True
    )
    print("matched point number", np.sum(match_result["matches"] > -1))

    best_color = colors[best_idx]
    best_depth = depths[best_idx]
    best_pose = camera_poses[best_idx].cpu().numpy()
    return best_color, best_depth, best_pose, match_result, camera_intrinsics


def registration_pnp(mesh_matching_points, raw_matching_points, intrinsic):
    # Solve the PNP and verify the reprojection error
    success, rvec, tvec = cv2.solvePnP(
        np.float32(mesh_matching_points),
        np.float32(raw_matching_points),
        np.float32(intrinsic),
        distCoeffs=np.zeros(4, dtype=np.float32),
        flags=cv2.SOLVEPNP_EPNP,
    )
    assert success, "solvePnP failed"
    projected_points, _ = cv2.projectPoints(
        np.float32(mesh_matching_points),
        rvec,
        tvec,
        intrinsic,
        np.zeros(4, dtype=np.float32),
    )
    error = np.linalg.norm(
        np.float32(raw_matching_points) - projected_points.reshape(-1, 2), axis=1
    ).mean()
    print(f"Reprojection Error: {error}")
    if error > 50:
        print(f"solvePnP failed for this case {case_name}.$$$$$$$$$$$$$$$$$$$$$$$$$$")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    mesh2raw_camera = np.eye(4, dtype=np.float32)
    mesh2raw_camera[:3, :3] = rotation_matrix
    mesh2raw_camera[:3, 3] = tvec.squeeze()

    return mesh2raw_camera


def registration_scale(mesh_matching_points_cam, matching_points_cam):
    # After PNP, optimize the scale in the camera coordinate
    def objective(scale, mesh_points, pcd_points):
        transformed_points = scale * mesh_points
        loss = np.sum(np.sum((transformed_points - pcd_points) ** 2, axis=1))
        return loss

    initial_scale = 1
    result = minimize(
        objective,
        initial_scale,
        args=(mesh_matching_points_cam, matching_points_cam),
        method="L-BFGS-B",
    )
    optimal_scale = result.x[0]
    print("Rescale:", optimal_scale)
    return optimal_scale


def deform_ARAP(initial_mesh_world, mesh_matching_points_world, matching_points):
    # Do the ARAP deformation based on the matching keypoints
    mesh_vertices = np.asarray(initial_mesh_world.vertices)
    kdtree = KDTree(mesh_vertices)
    _, mesh_points_indices = kdtree.query(mesh_matching_points_world)
    mesh_points_indices = np.asarray(mesh_points_indices, dtype=np.int32)
    deform_mesh = initial_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(mesh_points_indices),
        o3d.utility.Vector3dVector(matching_points),
        max_iter=1,
    )
    return deform_mesh, mesh_points_indices


def get_matching_ray_registration(
    mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
):
    # Get the matching indices and targets based on the viewpoint
    obs_points_cam = np.dot(
        w2c,
        np.hstack((obs_points_world, np.ones((obs_points_world.shape[0], 1)))).T,
    ).T
    obs_points_cam = obs_points_cam[:, :3]
    vertices_cam = np.dot(
        w2c,
        np.hstack(
            (
                np.asarray(mesh_world.vertices),
                np.ones((np.asarray(mesh_world.vertices).shape[0], 1)),
            )
        ).T,
    ).T
    vertices_cam = vertices_cam[:, :3]

    obs_kd = KDTree(obs_points_cam)

    new_indices = []
    new_targets = []
    # trimesh used to do the ray-casting test
    mesh.vertices = np.asarray(vertices_cam)[trimesh_indices]
    for index, vertex in enumerate(vertices_cam):
        ray_origins = np.array([[0, 0, 0]])
        ray_direction = vertex
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        ray_directions = np.array([ray_direction])
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False
        )

        ignore_flag = False

        if len(locations) > 0:
            first_intersection = locations[0]
            vertex_distance = np.linalg.norm(vertex)
            intersection_distance = np.linalg.norm(first_intersection)
            if intersection_distance < vertex_distance - 1e-4:
                # If the intersection point is not the vertex, it means the vertex is not visible from the camera viewpoint
                ignore_flag = True

        if ignore_flag:
            continue
        else:
            # Select the closest point to the ray of the observation points as the matching point
            indices = obs_kd.query_ball_point(vertex, 0.02)
            line_distances = line_point_distance(vertex, obs_points_cam[indices])
            # Get the closest point
            if len(line_distances) > 0:
                closest_index = np.argmin(line_distances)
                target = np.dot(
                    c2w, np.hstack((obs_points_cam[indices][closest_index], 1))
                )
                new_indices.append(index)
                new_targets.append(target[:3])

    new_indices = np.asarray(new_indices)
    new_targets = np.asarray(new_targets)

    return new_indices, new_targets


def deform_ARAP_ray_registration(
    deform_kp_mesh_world,
    obs_points_world,
    mesh,
    trimesh_indices,
    c2ws,
    w2cs,
    mesh_points_indices,
    matching_points,
):
    final_indices = []
    final_targets = []
    for index, target in zip(mesh_points_indices, matching_points):
        if index not in final_indices:
            final_indices.append(index)
            final_targets.append(target)

    for c2w, w2c in zip(c2ws, w2cs):
        new_indices, new_targets = get_matching_ray_registration(
            deform_kp_mesh_world, obs_points_world, mesh, trimesh_indices, c2w, w2c
        )
        for index, target in zip(new_indices, new_targets):
            if index not in final_indices:
                final_indices.append(index)
                final_targets.append(target)

    # Also need to adjust the positions to make sure they are above the table
    indices = np.where(np.asarray(deform_kp_mesh_world.vertices)[:, 2] > 0)[0]
    for index in indices:
        if index not in final_indices:
            final_indices.append(index)
            target = np.asarray(deform_kp_mesh_world.vertices)[index].copy()
            target[2] = 0
            final_targets.append(target)
        else:
            target = final_targets[final_indices.index(index)]
            if target[2] > 0:
                target[2] = 0
                final_targets[final_indices.index(index)] = target

    final_mesh_world = deform_kp_mesh_world.deform_as_rigid_as_possible(
        o3d.utility.IntVector(final_indices),
        o3d.utility.Vector3dVector(final_targets),
        max_iter=1,
    )
    return final_mesh_world


def line_point_distance(p, points):
    # Compute the distance between points and the line between p and [0, 0, 0]
    p = p / np.linalg.norm(p)
    points_to_origin = points
    cross_product = np.linalg.norm(np.cross(points_to_origin, p), axis=1)
    return cross_product / np.linalg.norm(p)


if __name__ == "__main__":
    existDir(output_dir)

    cam_idx = 0
    img_path = f"{base_path}/{case_name}/color/{cam_idx}/0.png"
    mesh_path = f"{base_path}/{case_name}/shape/object.glb"
    # Get the mask index of the object
    with open(f"{base_path}/{case_name}/mask/mask_info_{cam_idx}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_img_path = f"{base_path}/{case_name}/mask/{cam_idx}/{obj_idx}/0.png"
    # Load the metadata
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsic = np.array(data["intrinsics"])[cam_idx]

    # Load the c2w for the camera
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
        c2w = c2ws[cam_idx]
        w2c = np.linalg.inv(c2w)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]

    # Load the shape prior
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    mesh = as_mesh(mesh)

    # Load and process the image to get a cropped version for easy superglue
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    # Get mask bounding box, larger than the original bounding box
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # Calculate camera parameters
    fov = 2 * np.arctan(raw_img.shape[1] / (2 * intrinsic[0, 0]))

    if not os.path.exists(f"{output_dir}/best_match.pkl"):
        # 2D feature Matching to get the best pose of the object
        bbox = np.argwhere(mask_img > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            int(center[0] - size // 2),
            int(center[1] - size // 2),
            int(center[0] + size // 2),
            int(center[1] + size // 2),
        )
        # Make sure the bounding box is within the image
        bbox = (
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(raw_img.shape[1], bbox[2]),
            min(raw_img.shape[0], bbox[3]),
        )
        # Get the masked cropped image used for superglue
        crop_img = raw_img.copy()
        mask_bool = mask_img > 0
        crop_img[~mask_bool] = 0
        crop_img = crop_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)

        # Render the object and match the features
        best_color, best_depth, best_pose, match_result, camera_intrinsics = (
            pose_selection_render_superglue(
                raw_img,
                fov,
                mesh_path,
                mesh,
                crop_img,
                output_dir=output_dir,
            )
        )
        with open(f"{output_dir}/best_match.pkl", "wb") as f:
            pickle.dump(
                [
                    best_color,
                    best_depth,
                    best_pose,
                    match_result,
                    camera_intrinsics,
                    bbox,
                ],
                f,
            )
    else:
        with open(f"{output_dir}/best_match.pkl", "rb") as f:
            best_color, best_depth, best_pose, match_result, camera_intrinsics, bbox = (
                pickle.load(f)
            )

    # Process to get the matching points on the mesh and on the image
    # Get the projected 3D matching points on the mesh
    valid_matches = match_result["matches"] > -1
    render_matching_points = match_result["keypoints0"][valid_matches]
    mesh_matching_points, valid_mask = project_2d_to_3d(
        render_matching_points, best_depth, camera_intrinsics, best_pose
    )
    render_matching_points = render_matching_points[valid_mask]
    # Get the matching points on the raw image
    raw_matching_points_box = match_result["keypoints1"][
        match_result["matches"][valid_matches]
    ]
    raw_matching_points_box = raw_matching_points_box[valid_mask]
    raw_matching_points = raw_matching_points_box + np.array([bbox[0], bbox[1]])

    if VIS:
        # Do visualization for the matching
        plot_mesh_with_points(
            mesh,
            mesh_matching_points,
            f"{output_dir}/mesh_matching.png",
        )
        plot_image_with_points(
            best_depth,
            render_matching_points,
            f"{output_dir}/render_matching.png",
        )
        plot_image_with_points(
            raw_img,
            raw_matching_points,
            f"{output_dir}/raw_matching.png",
        )

    # Do PnP optimization to optimize the rotation between the 3D mesh keypoints and the 2D image keypoints
    mesh2raw_camera = registration_pnp(
        mesh_matching_points, raw_matching_points, intrinsic
    )

    if VIS:
        pnp_camera_pose = np.eye(4, dtype=np.float32)
        pnp_camera_pose[:3, :3] = np.linalg.inv(mesh2raw_camera[:3, :3])
        pnp_camera_pose[3, :3] = mesh2raw_camera[:3, 3]
        pnp_camera_pose[:, :2] = -pnp_camera_pose[:, :2]
        color, depth = render_image(
            mesh_path, pnp_camera_pose, raw_img.shape[1], raw_img.shape[0], fov, "cuda"
        )
        vis_mask = depth > 0
        color[0][~vis_mask] = raw_img[~vis_mask]
        plt.imsave(f"{output_dir}/pnp_results.png", color[0])

    # Transform the mesh into the real world coordinate
    mesh_matching_points_cam = np.dot(
        mesh2raw_camera,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_cam = mesh_matching_points_cam[:, :3]

    # Load the pcd in world coordinate of raw image matching points
    obs_points = []
    obs_colors = []
    pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
    mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
    data = np.load(pcd_path)
    with open(mask_path, "rb") as f:
        processed_masks = pickle.load(f)
    for i in range(3):
        points = data["points"][i]
        colors = data["colors"][i]
        mask = processed_masks[0][i]["object"]
        obs_points.append(points[mask])
        obs_colors.append(colors[mask])
        if i == 0:
            first_points = points
            first_mask = mask

    obs_points = np.vstack(obs_points)
    obs_colors = np.vstack(obs_colors)

    # Find the cloest points for the raw_matching_points
    new_match, matching_points = select_point(
        first_points, raw_matching_points, first_mask
    )
    matching_points_cam = np.dot(
        w2c, np.hstack((matching_points, np.ones((matching_points.shape[0], 1)))).T
    ).T
    matching_points_cam = matching_points_cam[:, :3]

    if VIS:
        # Draw the raw_matching_points and new matching points on the masked
        vis_img = raw_img.copy()
        vis_img[~first_mask] = 0
        plot_image_with_points(
            vis_img,
            raw_matching_points,
            f"{output_dir}/raw_matching_valid.png",
            new_match,
        )

    # Use the matching points in the camera coordinate to optimize the scame between the mesh and the observation
    optimal_scale = registration_scale(mesh_matching_points_cam, matching_points_cam)

    # Compute the rigid transformation from the original mesh to the final world coordinate
    scale_matrix = np.eye(4) * optimal_scale
    scale_matrix[3, 3] = 1
    mesh2world = np.dot(c2w, np.dot(scale_matrix, mesh2raw_camera))

    mesh_matching_points_world = np.dot(
        mesh2world,
        np.hstack(
            (mesh_matching_points, np.ones((mesh_matching_points.shape[0], 1)))
        ).T,
    ).T
    mesh_matching_points_world = mesh_matching_points_world[:, :3]

    # Do the ARAP based on the matching keypoints
    # Convert the mesh to open3d to use the ARAP function
    initial_mesh_world = o3d.geometry.TriangleMesh()
    initial_mesh_world.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    initial_mesh_world.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    # Need to remove the duplicated vertices to enable open3d, however, the duplicated points are important in trimesh for texture
    initial_mesh_world = initial_mesh_world.remove_duplicated_vertices()
    # Get the index from original vertices to the mesh vertices, mapping between trimesh and open3d
    kdtree = KDTree(initial_mesh_world.vertices)
    _, trimesh_indices = kdtree.query(np.asarray(mesh.vertices))
    trimesh_indices = np.asarray(trimesh_indices, dtype=np.int32)
    initial_mesh_world.transform(mesh2world)

    # ARAP based on the keypoints
    deform_kp_mesh_world, mesh_points_indices = deform_ARAP(
        initial_mesh_world, mesh_matching_points_world, matching_points
    )

    # Do the ARAP based on both the ray-casting matching and the keypoints
    # Identify the vertex which blocks or blocked by the observation, then match them with the observation points on the ray
    final_mesh_world = deform_ARAP_ray_registration(
        deform_kp_mesh_world,
        obs_points,
        mesh,
        trimesh_indices,
        c2ws,
        w2cs,
        mesh_points_indices,
        matching_points,
    )

    if VIS:
        final_mesh_world.compute_vertex_normals()

        # Visualize the partial observation and the mesh
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)

        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Render the final stuffs as a turntable video
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        height, width, _ = dummy_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_writer = cv2.VideoWriter(
            f"{output_dir}/final_matching.mp4", fourcc, 30, (width, height)
        )
        # final_mesh_world.compute_vertex_normals()
        # final_mesh_world.translate([0, 0, 0.2])
        # mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(final_mesh_world)
        # o3d.visualization.draw_geometries([pcd, final_mesh_world], window_name="Matching")
        vis.add_geometry(pcd)
        vis.add_geometry(final_mesh_world)
        # vis.add_geometry(coordinate)
        view_control = vis.get_view_control()

        for j in range(360):
            view_control.rotate(10, 0)
            vis.poll_events()
            vis.update_renderer()
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        vis.destroy_window()

    mesh.vertices = np.asarray(final_mesh_world.vertices)[trimesh_indices]
    mesh.export(f"{output_dir}/final_mesh.glb")
```

data_process/data_process_mask.py
```python
# Process the mask data to filter out the outliers and generate the processed masks

import numpy as np
import open3d as o3d
import json
from tqdm import tqdm
import os
import glob
import cv2
import pickle
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--controller_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
CONTROLLER_NAME = args.controller_name

processed_masks = {}


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def process_pcd_mask(frame_idx, pcd_path, mask_path, mask_info, num_cam):
    global processed_masks
    processed_masks[frame_idx] = {}

    # Load the pcd data
    data = np.load(f"{pcd_path}/{frame_idx}.npz")
    points = data["points"]
    colors = data["colors"]
    masks = data["masks"]

    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()

    for i in range(num_cam):
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        object_colors = colors[i][object_mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points)
        pcd.colors = o3d.utility.Vector3dVector(object_colors)
        object_pcd += pcd

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        controller_colors = colors[i][controller_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points)
        pcd.colors = o3d.utility.Vector3dVector(controller_colors)
        controller_pcd += pcd

    # Apply the outlier removal
    cl, ind = object_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_object_points = np.asarray(
        object_pcd.select_by_index(ind, invert=True).points
    )
    object_pcd = object_pcd.select_by_index(ind)

    cl, ind = controller_pcd.remove_radius_outlier(nb_points=40, radius=0.01)
    filtered_controller_points = np.asarray(
        controller_pcd.select_by_index(ind, invert=True).points
    )
    controller_pcd = controller_pcd.select_by_index(ind)

    # controller_pcd.paint_uniform_color([1, 0, 0])
    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])
    object_pcd = o3d.geometry.PointCloud()
    controller_pcd = o3d.geometry.PointCloud()
    for i in range(num_cam):
        processed_masks[frame_idx][i] = {}
        # Load the object mask
        object_idx = mask_info[i]["object"]
        mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
        object_mask = np.logical_and(masks[i], mask)
        object_points = points[i][object_mask]
        indices = np.nonzero(object_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the object_points in the filtered points
        object_indices = []
        for j, point in enumerate(object_points):
            if tuple(point) in filtered_object_points:
                object_indices.append(j)
        original_indices = [indices_list[j] for j in object_indices]
        # Update the object mask
        for idx in original_indices:
            object_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["object"] = object_mask

        # Load the controller mask
        controller_mask = np.zeros_like(masks[i])
        for controller_idx in mask_info[i]["controller"]:
            mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
            controller_mask = np.logical_or(controller_mask, mask)
        controller_mask = np.logical_and(masks[i], controller_mask)
        controller_points = points[i][controller_mask]
        indices = np.nonzero(controller_mask)
        indices_list = list(zip(indices[0], indices[1]))
        # Locate all the controller_points in the filtered points
        controller_indices = []
        for j, point in enumerate(controller_points):
            if tuple(point) in filtered_controller_points:
                controller_indices.append(j)
        original_indices = [indices_list[j] for j in controller_indices]
        # Update the controller mask
        for idx in original_indices:
            controller_mask[idx[0], idx[1]] = 0
        processed_masks[frame_idx][i]["controller"] = controller_mask

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][object_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][object_mask])

        object_pcd += pcd

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i][controller_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[i][controller_mask])

        controller_pcd += pcd

    # o3d.visualization.draw_geometries([object_pcd, controller_pcd])

    return object_pcd, controller_pcd


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))
    # Load the mask metadata
    mask_info = {}
    for i in range(num_cam):
        with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
            data = json.load(f)
        mask_info[i] = {}
        for key, value in data.items():
            if value != CONTROLLER_NAME:
                if "object" in mask_info[i]:
                    # TODO: Handle the case when there are multiple objects
                    import pdb
                    pdb.set_trace()
                mask_info[i]["object"] = int(key)
            if value == CONTROLLER_NAME:
                if "controller" in mask_info[i]:
                    mask_info[i]["controller"].append(int(key))
                else:
                    mask_info[i]["controller"] = [int(key)]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    object_pcd = None
    controller_pcd = None
    for i in tqdm(range(frame_num)):
        temp_object_pcd, temp_controller_pcd = process_pcd_mask(
            i, pcd_path, mask_path, mask_info, num_cam
        )
        if i == 0:
            object_pcd = temp_object_pcd
            controller_pcd = temp_controller_pcd
            vis.add_geometry(object_pcd)
            vis.add_geometry(controller_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            object_pcd.points = o3d.utility.Vector3dVector(temp_object_pcd.points)
            object_pcd.colors = o3d.utility.Vector3dVector(temp_object_pcd.colors)
            controller_pcd.points = o3d.utility.Vector3dVector(
                temp_controller_pcd.points
            )
            controller_pcd.colors = o3d.utility.Vector3dVector(
                temp_controller_pcd.colors
            )
            vis.update_geometry(object_pcd)
            vis.update_geometry(controller_pcd)
            vis.poll_events()
            vis.update_renderer()

    # Save the processed masks considering both depth filter, semantic filter and outlier filter
    with open(f"{base_path}/{case_name}/mask/processed_masks.pkl", "wb") as f:
        pickle.dump(processed_masks, f)

    # Deprecated for now
    # # Generate the videos with for masked objects and controllers
    # exist_dir(f"{base_path}/{case_name}/temp_mask")
    # for i in range(num_cam):
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/object")
    #     exist_dir(f"{base_path}/{case_name}/temp_mask/{i}/controller")
    #     object_idx = mask_info[i]["object"]
    #     for frame_idx in range(frame_num):
    #         object_mask = read_mask(f"{mask_path}/{i}/{object_idx}/{frame_idx}.png")
    #         img = cv2.imread(f"{base_path}/{case_name}/color/{i}/{frame_idx}.png")
    #         masked_object_img = cv2.bitwise_and(
    #             img, img, mask=object_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/object/{frame_idx}.png",
    #             masked_object_img,
    #         )

    #         controller_mask = np.zeros_like(object_mask)
    #         for controller_idx in mask_info[i]["controller"]:
    #             mask = read_mask(f"{mask_path}/{i}/{controller_idx}/{frame_idx}.png")
    #             controller_mask = np.logical_or(controller_mask, mask)
    #         masked_controller_img = cv2.bitwise_and(
    #             img, img, mask=controller_mask.astype(np.uint8) * 255
    #         )
    #         cv2.imwrite(
    #             f"{base_path}/{case_name}/temp_mask/{i}/controller/{frame_idx}.png",
    #             masked_controller_img,
    #         )

    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/object/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/object_{i}.mp4"
    #     )
    #     os.system(
    #         f"ffmpeg -r 30 -start_number 0 -f image2 -i {base_path}/{case_name}/temp_mask/{i}/controller/%d.png -vcodec libx264 -crf 0  -pix_fmt yuv420p {base_path}/{case_name}/temp_mask/controller_{i}.mp4"
    #     )
```

data_process/data_process_pcd.py
```python
# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/model/pcdHelper.py
def getPcdFromDepth(
    depth,
    intrinsic,
):
    # Depth in meters
    height, width = np.shape(depth)

    # Reshape the depth array to invert the depth values
    depth = -depth

    # Create a grid of (x, y) coordinates
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # Create a meshgrid for x and y coordinates
    X, Y = np.meshgrid(x_coords, y_coords)

    # Calculate points using vectorized operations
    old_points = np.stack([(width - X) * depth, Y * depth, depth], axis=-1)

    # Flatten the old_points array and calculate the new points using matrix multiplication
    points = np.dot(np.linalg.inv(intrinsic), old_points.reshape(-1, 3).T).T.reshape(
        old_points.shape
    )

    points[:, :, 1] *= -1
    points[:, :, 2] *= -1

    return points


def get_pcd_from_data(path, frame_idx, num_cam, intrinsics, c2ws):
    total_points = []
    total_colors = []
    total_masks = []
    for i in range(num_cam):
        color = cv2.imread(f"{path}/color/{i}/{frame_idx}.png")
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth = np.load(f"{path}/depth/{i}/{frame_idx}.npy") / 1000.0

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )
        masks = np.logical_and(points[:, :, 2] > 0.2, points[:, :, 2] < 1.5)
        points_flat = points.reshape(-1, 3)
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)
    # pcd = o3d.geometry.PointCloud()
    # visualize_points = []
    # visualize_colors = []
    # for i in range(num_cam):
    #     visualize_points.append(
    #         total_points[i][total_masks[i]].reshape(-1, 3)
    #     )
    #     visualize_colors.append(
    #         total_colors[i][total_masks[i]].reshape(-1, 3)
    #     )
    # visualize_points = np.concatenate(visualize_points)
    # visualize_colors = np.concatenate(visualize_colors)
    # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
    # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
    # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
    # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
    # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
    # visualize_points = visualize_points[mask]
    # visualize_colors = visualize_colors[mask]
        
    # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])
    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    frame_num = data["frame_num"]
    print(data["serial_numbers"])

    num_cam = len(intrinsics)
    c2ws = pickle.load(open(f"{base_path}/{case_name}/calibrate.pkl", "rb"))

    exist_dir(f"{base_path}/{case_name}/pcd")

    cameras = []
    # Visualize the cameras
    for i in range(num_cam):
        camera = getCamera(
            c2ws[i],
            intrinsics[i, 0, 0],
            intrinsics[i, 1, 1],
            intrinsics[i, 0, 2],
            intrinsics[i, 1, 2],
            z_flip=True,
            scale=0.2,
        )
        cameras += camera

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for camera in cameras:
        vis.add_geometry(camera)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coordinate)

    pcd = None
    for i in tqdm(range(frame_num)):
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, num_cam, intrinsics, c2ws
        )

        if i == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            vis.add_geometry(pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )
```

data_process/data_process_sample.py
```python
# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points

import numpy as np
import open3d as o3d
import pickle
import matplotlib.pyplot as plt
import trimesh
import cv2
from utils.align_util import as_mesh
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.005)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def process_unique_points(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Get the unique index in the object points
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    # Make sure all points are above the ground
    object_points[object_points[..., 2] > 0, 2] = 0

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        # Sample the surface points
        surface_points, _ = trimesh.sample.sample_surface(
            trimesh_mesh, num_surface_points
        )
        # Sample the interior points
        interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)

    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    if SHAPE_PRIOR:
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    # Render the final pcd with interior filling as a turntable video
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
    )

    vis.add_geometry(all_pcd)
    # vis.add_geometry(coorindate)
    view_control = vis.get_view_control()
    for j in range(360):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    vis.destroy_window()

    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)

    track_data = process_unique_points(track_data)

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)
```

data_process/data_process_track.py
```python
# FIlter the tracking based on the object and controller mask, filter the track based on the neighbour motion
# Get the nearest controller points that are valid across all frames

import numpy as np
import open3d as o3d
from tqdm import tqdm
import os
import glob
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


# Based on the valid mask, filter out the bad tracking data
def filter_track(track_path, pcd_path, mask_path, frame_num, num_cam):
    with open(f"{mask_path}/processed_masks.pkl", "rb") as f:
        processed_masks = pickle.load(f)

    # Filter out the points not valid in the first frame
    object_points = []
    object_colors = []
    object_visibilities = []
    controller_points = []
    controller_colors = []
    controller_visibilities = []
    for i in range(num_cam):
        current_track_data = np.load(f"{track_path}/{i}.npz")
        # Filter out the track data
        tracks = current_track_data["tracks"]
        tracks = np.round(tracks).astype(int)
        visibility = current_track_data["visibility"]
        assert tracks.shape[0] == frame_num
        num_points = np.shape(tracks)[1]

        # Locate the track points in the object mask of the first frame
        object_mask = processed_masks[0][i]["object"]
        track_object_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_object_idx[j] = object_mask[tracks[0, j, 0], tracks[0, j, 1]]
        # Locate the controller points in the controller mask of the first frame
        controller_mask = processed_masks[0][i]["controller"]
        track_controller_idx = np.zeros((num_points), dtype=int)
        for j in range(num_points):
            if visibility[0, j] == 1:
                track_controller_idx[j] = controller_mask[
                    tracks[0, j, 0], tracks[0, j, 1]
                ]

        # Filter out bad tracking in other frames
        for frame_idx in range(1, frame_num):
            # Filter based on object_mask
            object_mask = processed_masks[frame_idx][i]["object"]
            for j in range(num_points):
                try:
                    if track_object_idx[j] == 1 and visibility[frame_idx, j] == 1:
                        if not object_mask[
                            tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                        ]:
                            visibility[frame_idx, j] = 0
                except:
                    # Sometimes the track coordinate is out of image
                    visibility[frame_idx, j] = 0
            # Filter based on controller_mask
            controller_mask = processed_masks[frame_idx][i]["controller"]
            for j in range(num_points):
                if track_controller_idx[j] == 1 and visibility[frame_idx, j] == 1:
                    if not controller_mask[
                        tracks[frame_idx, j, 0], tracks[frame_idx, j, 1]
                    ]:
                        visibility[frame_idx, j] = 0

        # Get the track point cloud
        track_points = np.zeros((frame_num, num_points, 3))
        track_colors = np.zeros((frame_num, num_points, 3))
        for frame_idx in range(frame_num):
            data = np.load(f"{pcd_path}/{frame_idx}.npz")
            points = data["points"]
            colors = data["colors"]

            track_points[frame_idx][np.where(visibility[frame_idx])] = points[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]
            track_colors[frame_idx][np.where(visibility[frame_idx])] = colors[i][
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 0],
                tracks[frame_idx, np.where(visibility[frame_idx])[0], 1],
            ]

        object_points.append(track_points[:, np.where(track_object_idx)[0], :])
        object_colors.append(track_colors[:, np.where(track_object_idx)[0], :])
        object_visibilities.append(visibility[:, np.where(track_object_idx)[0]])
        controller_points.append(track_points[:, np.where(track_controller_idx)[0], :])
        controller_colors.append(track_colors[:, np.where(track_controller_idx)[0], :])
        controller_visibilities.append(visibility[:, np.where(track_controller_idx)[0]])

    object_points = np.concatenate(object_points, axis=1)
    object_colors = np.concatenate(object_colors, axis=1)
    object_visibilities = np.concatenate(object_visibilities, axis=1)
    controller_points = np.concatenate(controller_points, axis=1)
    controller_colors = np.concatenate(controller_colors, axis=1)
    controller_visibilities = np.concatenate(controller_visibilities, axis=1)

    track_data = {}
    track_data["object_points"] = object_points
    track_data["object_colors"] = object_colors
    track_data["object_visibilities"] = object_visibilities
    track_data["controller_points"] = controller_points
    track_data["controller_colors"] = controller_colors
    track_data["controller_visibilities"] = controller_visibilities

    return track_data


def filter_motion(track_data, neighbor_dist=0.01):
    # Calculate the motion of each point
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions = np.zeros_like(object_points)
    object_motions[:-1] = object_points[1:] - object_points[:-1]
    object_motions_valid = np.zeros_like(object_visibilities)
    object_motions_valid[:-1] = np.logical_and(
        object_visibilities[:-1], object_visibilities[1:]
    )

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    num_frames = object_points.shape[0]
    num_points = object_points.shape[1]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points[i])
        pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # modified_points = []
        # new_points = []
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if object_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                object_points[i, j], neighbor_dist
            )
            neighbors = [index for index in idx if object_motions_valid[i, index] == 1]
            if len(neighbors) < 5:
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])
            motion_diff = np.linalg.norm(
                object_motions[i, j] - object_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                object_motions_valid[i, j] = 0
                # modified_points.append(object_points[i, j])
                # new_points.append(object_points[i + 1, j])

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            object_points[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            object_colors[i][np.where(object_motions_valid[i])]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])]
        )

        # modified_pcd = o3d.geometry.PointCloud()
        # modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
        # modified_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
        # )

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = o3d.utility.Vector3dVector(new_points)
        # new_pcd.colors = o3d.utility.Vector3dVector(
        #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
        # )
        if i == 0:
            render_motion_pcd = motion_pcd
            # render_modified_pcd = modified_pcd
            # render_new_pcd = new_pcd
            vis.add_geometry(render_motion_pcd)
            # vis.add_geometry(render_modified_pcd)
            # vis.add_geometry(render_new_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            # render_modified_pcd.points = o3d.utility.Vector3dVector(modified_points)
            # render_modified_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([1, 0, 0]) * np.ones((len(modified_points), 3))
            # )
            # render_new_pcd.points = o3d.utility.Vector3dVector(new_points)
            # render_new_pcd.colors = o3d.utility.Vector3dVector(
            #     np.array([0, 1, 0]) * np.ones((len(new_points), 3))
            # )
            vis.update_geometry(render_motion_pcd)
            # vis.update_geometry(render_modified_pcd)
            # vis.update_geometry(render_new_pcd)
            vis.poll_events()
            vis.update_renderer()
        # modified_num = len(modified_points)
        # print(f"Object Frame {i}: {modified_num} points are modified")

    vis.destroy_window()
    track_data["object_motions_valid"] = object_motions_valid

    controller_points = track_data["controller_points"]
    controller_colors = track_data["controller_colors"]
    controller_visibilities = track_data["controller_visibilities"]
    controller_motions = np.zeros_like(controller_points)
    controller_motions[:-1] = controller_points[1:] - controller_points[:-1]
    controller_motions_valid = np.zeros_like(controller_visibilities)
    controller_motions_valid[:-1] = np.logical_and(
        controller_visibilities[:-1], controller_visibilities[1:]
    )
    num_points = controller_points.shape[1]
    # Filter all points that disappear in the sequence
    mask = np.prod(controller_visibilities, axis=0)

    y_min, y_max = np.min(controller_points[0, :, 1]), np.max(
        controller_points[0, :, 1]
    )
    y_normalized = (controller_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for i in tqdm(range(num_frames - 1)):
        # Convert the points of the current frame to an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(controller_points[i])
        pcd.colors = o3d.utility.Vector3dVector(controller_colors[i])
        # Build the KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        # Get the neighbors for each points and filter motion based on the motion difference between neighbours and the point
        for j in range(num_points):
            if mask[j] == 0:
                controller_motions_valid[i, j] = 0
            if controller_motions_valid[i, j] == 0:
                continue
            # Get the neighbors within neighbor_dist
            [k, idx, _] = kdtree.search_radius_vector_3d(
                controller_points[i, j], neighbor_dist
            )
            neighbors = [
                index for index in idx if controller_motions_valid[i, index] == 1
            ]
            if len(neighbors) < 5:
                controller_motions_valid[i, j] = 0
                mask[j] = 0

            motion_diff = np.linalg.norm(
                controller_motions[i, j] - controller_motions[i, neighbors], axis=1
            )
            if (motion_diff < neighbor_dist / 2).sum() < 0.5 * len(neighbors):
                controller_motions_valid[i, j] = 0
                mask[j] = 0

        motion_pcd = o3d.geometry.PointCloud()
        motion_pcd.points = o3d.utility.Vector3dVector(
            controller_points[i][np.where(mask)]
        )
        motion_pcd.colors = o3d.utility.Vector3dVector(
            controller_colors[i][np.where(controller_motions_valid[i])]
        )

        if i == 0:
            render_motion_pcd = motion_pcd
            vis.add_geometry(render_motion_pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_motion_pcd.points = o3d.utility.Vector3dVector(motion_pcd.points)
            render_motion_pcd.colors = o3d.utility.Vector3dVector(motion_pcd.colors)
            vis.update_geometry(render_motion_pcd)
            vis.poll_events()
            vis.update_renderer()

    track_data["controller_mask"] = mask
    return track_data


def get_final_track_data(track_data, controller_threhsold=0.01):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]
    mask = track_data["controller_mask"]

    new_controller_points = controller_points[:, np.where(mask)[0], :]
    assert len(new_controller_points[0]) >= 30
    # Do farthest point sampling on the valid controller points to select the final controller points
    valid_indices = np.arange(len(new_controller_points[0]))
    points_map = {}
    sample_points = []
    for i in valid_indices:
        points_map[tuple(new_controller_points[0, i])] = i
        sample_points.append(new_controller_points[0, i])
    sample_points = np.array(sample_points)
    sample_pcd = o3d.geometry.PointCloud()
    sample_pcd.points = o3d.utility.Vector3dVector(sample_points)
    fps_pcd = sample_pcd.farthest_point_down_sample(30)
    final_indices = []
    for point in fps_pcd.points:
        final_indices.append(points_map[tuple(point)])

    print(f"Controller Point Number: {len(final_indices)}")

    # Get the nearest controller points and their colors
    nearest_controller_points = new_controller_points[:, final_indices]

    # object_pcd = o3d.geometry.PointCloud()
    # object_pcd.points = o3d.utility.Vector3dVector(valid_object_points)
    # object_pcd.colors = o3d.utility.Vector3dVector(
    #     object_colors[0][np.where(object_motions_valid[0])]
    # )
    # controller_meshes = []
    # for j in range(nearest_controller_points.shape[1]):
    #     origin = nearest_controller_points[0, j]
    #     origin_color = [1, 0, 0]
    #     controller_meshes.append(
    #         getSphereMesh(origin, color=origin_color, radius=0.005)
    #     )
    # o3d.visualization.draw_geometries([object_pcd])
    # o3d.visualization.draw_geometries([object_pcd] + controller_meshes)

    track_data.pop("controller_points")
    track_data.pop("controller_colors")
    track_data.pop("controller_visibilities")
    track_data["controller_points"] = nearest_controller_points

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_motions_valid[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_motions_valid[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
            vis.poll_events()
            vis.update_renderer()


if __name__ == "__main__":
    pcd_path = f"{base_path}/{case_name}/pcd"
    mask_path = f"{base_path}/{case_name}/mask"
    track_path = f"{base_path}/{case_name}/cotracker"

    num_cam = len(glob.glob(f"{mask_path}/mask_info_*.json"))
    frame_num = len(glob.glob(f"{pcd_path}/*.npz"))

    # Filter the track data using the semantic mask of object and controller
    track_data = filter_track(track_path, pcd_path, mask_path, frame_num, num_cam)
    # Filter motion
    track_data = filter_motion(track_data)
    # # Save the filtered track data
    # with open(f"test2.pkl", "wb") as f:
    #     pickle.dump(track_data, f)

    # with open(f"test2.pkl", "rb") as f:
    #     track_data = pickle.load(f)

    track_data = get_final_track_data(track_data)

    with open(f"{base_path}/{case_name}/track_process_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)
```

data_process/dense_track.py
```python
# Use co-tracker to track the ibject and controller in the video (pick 5000 pixels in the masked area)

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

num_cam = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == num_cam
device = "cuda"


def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")
        # Load the video
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        # Load the first-frame mask to get all query points from all masks
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # Revert x and y
        query_pixels = query_pixels[:, ::-1]
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        # Randomly select 5000 query points
        query_pixels = query_pixels[torch.randperm(query_pixels.shape[0])[:5000]]

        # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
        # pred_tracks, pred_visibility = cotracker(video, queries=query_pixels[None], backward_tracking=True)
        # pred_tracks, pred_visibility = cotracker(video, grid_query_frame=0)

        # # Run Online CoTracker:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        cotracker(video_chunk=video, is_first_step=True, queries=query_pixels[None])

        # Process the video
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )  # B T N 2,  B T N 1
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        vis.visualize(video, pred_tracks, pred_visibility, filename=f"{i}")
        # Save the tracking data into npz
        track_to_save = pred_tracks[0].cpu().numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].cpu().numpy()
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
```

data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py
```python
batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True```

data_process/image_upscale.py
```python
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
from argparse import ArgumentParser
import cv2
import numpy as np

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--mask_path", type=str, default=None)
parser.add_argument("--output_path", type=str)
parser.add_argument("--category", type=str)
args = parser.parse_args()

img_path = args.img_path
mask_path = args.mask_path
output_path = args.output_path
category = args.category


# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(
    model_id, torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

# let's download an  image
low_res_img = Image.open(img_path).convert("RGB")
if mask_path is not None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bbox = np.argwhere(mask > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    low_res_img = low_res_img.crop(bbox)  # type: ignore

prompt = f"Hand manipulates a {category}."

upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
upscaled_image.save(output_path)
```

data_process/match_pairs.py
```python
#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import sys
import os

sys.path.append(os.getcwd())
from models.matching import Matching
from models.utils import (
    make_matching_plot,
    AverageTimer,
    read_image,
)

torch.set_grad_enabled(False)


def image_pair_matching(
    input_images,
    ref_image,
    output_dir,
    resize=[-1],
    resize_float=False,
    superglue="indoor",
    max_keypoints=1024,
    keypoint_threshold=0.005,
    nms_radius=4,
    sinkhorn_iterations=20,
    match_threshold=0.2,
    viz=False,
    fast_viz=False,
    cache=True,
    show_keypoints=False,
    viz_extension="png",
    save=False,
    viz_best=True,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory "{}"'.format(output_dir))
    if viz:
        print('`Will writ`e visualization images to directory "{}"'.format(output_dir))

    timer = AverageTimer(newline=True)
    match_nums = []
    match_result = []

    best_match = {}
    best_match_num = -1

    for i, image in enumerate(input_images):
        matches_path = output_dir / "matches_{}.npz".format(i)
        viz_path = output_dir / "matches_{}.{}".format(i, viz_extension)

        do_match = True
        do_viz = viz
        if cache:
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError("Cannot load matches .npz file: %s" % matches_path)

                kpts0, kpts1 = results["keypoints0"], results["keypoints1"]
                matches, conf = results["matches"], results["match_confidence"]
                do_match = False
            if viz and viz_path.exists():
                do_viz = False
            timer.update("load_cache")

        rot0, rot1 = 0, 0
        image0, inp0, scales0 = read_image(image, device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(
            ref_image, device, resize, rot1, resize_float
        )
        if image0 is None or image1 is None:
            print("Problem reading image pair: {} and ref".format(i))
            exit(1)
        timer.update("load_image")

        if do_match:
            pred = matching({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            timer.update("matcher")

            out_matches = {
                "keypoints0": kpts0,
                "keypoints1": kpts1,
                "matches": matches,
                "match_confidence": conf,
            }
            match_result.append(out_matches)
            if save:
                np.savez(str(matches_path), **out_matches)
        else:
            match_result.append(results)

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        match_nums.append(len(mkpts0))

        if len(mkpts0) > best_match_num:
            best_match_num = len(mkpts0)
            best_match["image0"] = image0
            best_match["image1"] = image1
            best_match["kpts0"] = kpts0
            best_match["kpts1"] = kpts1
            best_match["mkpts0"] = mkpts0
            best_match["mkpts1"] = mkpts1
            best_match["mconf"] = mconf

        if do_viz:
            color = cm.jet(mconf)
            text = [
                "SuperGlue",
                "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
                "Matches: {}".format(len(mkpts0)),
            ]
            if rot0 != 0 or rot1 != 0:
                text.append("Rotation: {}:{}".format(rot0, rot1))

            k_thresh = matching.superpoint.config["keypoint_threshold"]
            m_thresh = matching.superglue.config["match_threshold"]
            small_text = [
                "Keypoint Threshold: {:.4f}".format(k_thresh),
                "Match Threshold: {:.2f}".format(m_thresh),
                "Image Pair: {} : ref".format(i),
            ]

            make_matching_plot(
                image0,
                image1,
                kpts0,
                kpts1,
                mkpts0,
                mkpts1,
                color,
                text,
                viz_path,
                show_keypoints,
                fast_viz,
                small_text,
            )

            timer.update("viz_match")
    best_pose = match_nums.index(max(match_nums))

    if viz_best:
        viz_path = f"{output_dir}/best_match.{viz_extension}"
        color = cm.jet(best_match["mconf"])
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(
                len(best_match["kpts0"]), len(best_match["kpts1"])
            ),
            "Matches: {}".format(len(best_match["mkpts0"])),
        ]

        make_matching_plot(
            best_match["image0"],
            best_match["image1"],
            best_match["kpts0"],
            best_match["kpts1"],
            best_match["mkpts0"],
            best_match["mkpts1"],
            color,
            text,
            viz_path,
            show_keypoints,
            fast_viz,
        )

        timer.update("viz_match")
    return best_pose, match_result[best_pose]
```

data_process/models/__init__.py
```python
```

data_process/models/matching.py
```python
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch

from .superpoint import SuperPoint
from .superglue import SuperGlue


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**pred, **self.superglue(data)}

        return pred
```

data_process/models/superglue.py
```python
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
        }
```

data_process/models/superpoint.py
```python
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        x = self.relu(self.conv1a(data['image']))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }
```

data_process/models/utils.py
```python
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)


def read_image(image, device, resize, rotation, resize_float):
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image, device)
    return image, inp, scales


# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, small_text=[]):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)
```

data_process/segment.py
```python
# Process to get the masks of the controller and the object
import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_num = 3
assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num
print(f"Processing {case_name}")

for camera_idx in range(camera_num):
    print(f"Processing {case_name} camera {camera_idx}")
    os.system(
        f"python ./data_process/segment_util_video.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
    )
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
```

data_process/segment_util_image.py
```python
import cv2
import torch
import numpy as np
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from argparse import ArgumentParser

"""
Hyper parameters
"""

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_path", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
args = parser.parse_args()

img_path = args.img_path
output_path = args.output_path
TEXT_PROMPT = args.TEXT_PROMPT

SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = (
    "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT = (
    "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino model
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
text = TEXT_PROMPT

image_source, image = load_image(img_path)

sam2_predictor.set_image(image_source)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=text,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

"""
Post-process the output of the model to get the masks, scores, and logits for visualization
"""
# convert the shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)


confidences = confidences.numpy().tolist()
class_names = labels

OBJECTS = class_names

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

print(f"Detected {len(masks)} objects")

raw_img = cv2.imread(img_path)
mask_img = (masks[0] * 255).astype(np.uint8)

ref_img = np.zeros((h, w, 4), dtype=np.uint8)
mask_bool = mask_img > 0
ref_img[mask_bool, :3] = raw_img[mask_bool]
ref_img[:, :, 3] = mask_bool.astype(np.uint8) * 255
cv2.imwrite(output_path, ref_img)
```

data_process/segment_util_video.py
```python
import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
import json
from argparse import ArgumentParser

"""
Hyperparam for Ground and Tracking
"""

# Put below base path into args
parser = ArgumentParser()
parser.add_argument("--base_path", type=str, default="/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/real_collect")
parser.add_argument("--case_name", type=str)
parser.add_argument("--TEXT_PROMPT", type=str)
parser.add_argument("--camera_idx", type=int)
parser.add_argument("--output_path", type=str, default="NONE")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
camera_idx = args.camera_idx
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"
else:
    output_path = args.output_path

def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"  # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_idx}.mp4"
existDir(f"{base_path}/{case_name}/tmp_data")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}")
existDir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}")

SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_idx}"

"""
Step 1: Environment settings and model initialization for Grounding DINO and SAM 2
"""
# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)


# init sam image predictor and video predictor model
sam2_checkpoint = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)


video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
print(video_info)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

# saving video to frames
source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for frame in tqdm(frame_generator, desc="Saving Video Frames"):
        sink.save_image(frame)

# scan all the JPEG frame names in this directory
frame_names = [
    p
    for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# init video predictor state
inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

ann_frame_idx = 0  # the frame index we interact with
"""
Step 2: Prompt Grounding DINO 1.5 with Cloud API for box coordinates
"""

# prompt grounding dino to get the box coordinates on specific frame
img_path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[ann_frame_idx])
image_source, image = load_image(img_path)

boxes, confidences, labels = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
)

# process the box prompt for SAM 2
h, w, _ = image_source.shape
boxes = boxes * torch.Tensor([w, h, w, h])
input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
confidences = confidences.numpy().tolist()
class_names = labels

print(input_boxes)

# prompt SAM image predictor to get the mask for the object
image_predictor.set_image(image_source)

# process the detection results
OBJECTS = class_names

print(OBJECTS)

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# prompt SAM 2 image predictor to get the mask for the object
masks, scores, logits = image_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)
# convert the mask shape to (n, H, W)
if masks.ndim == 4:
    masks = masks.squeeze(1)

"""
Step 3: Register each object's positive points to video predictor with seperate add_new_points call
"""

assert PROMPT_TYPE_FOR_VIDEO in [
    "point",
    "box",
    "mask",
], "SAM 2 video predictor only support point/box/mask prompt"

if PROMPT_TYPE_FOR_VIDEO == "box":
    for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes)):
        _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=object_id,
            box=box,
        )
else:
    raise NotImplementedError(
        "SAM 2 video predictor only support point/box/mask prompts"
    )

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
video_segments = {}  # video_segments contains the per-frame segmentation results
for (
    out_frame_idx,
    out_obj_ids,
    out_mask_logits,
) in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

"""
Step 5: Visualize the segment results across the video and save them
"""

existDir(f"{output_path}/mask/")
existDir(f"{output_path}/mask/{camera_idx}")

ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS)}

# Save the id_to_objects into json
with open(f"{output_path}/mask/mask_info_{camera_idx}.json", "w") as f:
    json.dump(ID_TO_OBJECTS, f)

for frame_idx, masks in video_segments.items():
    for obj_id, mask in masks.items():
        existDir(f"{output_path}/mask/{camera_idx}/{obj_id}")
        # mask is 1 * H * W
        Image.fromarray((mask[0] * 255).astype(np.uint8)).save(
            f"{output_path}/mask/{camera_idx}/{obj_id}/{frame_idx}.png"
        )
```

data_process/shape_prior.py
```python
import os

# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ["SPCONV_ALGO"] = "native"  # Can be 'native' or 'auto', default is 'auto'.
# 'auto' is faster but will do benchmarking at the beginning.
# Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--img_path",
    type=str,
)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

img_path = args.img_path
output_dir = args.output_dir

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

final_im = Image.open(img_path).convert("RGBA")
assert not np.all(np.array(final_im)[:, :, 3] == 255)

# Run the pipeline
outputs = pipeline.run(
    final_im,
)

video_gs = render_utils.render_video(outputs["gaussian"][0])["color"]
video_mesh = render_utils.render_video(outputs["mesh"][0])["normal"]
video = [
    np.concatenate([frame_gs, frame_mesh], axis=1)
    for frame_gs, frame_mesh in zip(video_gs, video_mesh)
]
imageio.mimsave(f"{output_dir}/visualization.mp4", video, fps=30)

# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs["gaussian"][0],
    outputs["mesh"][0],
    # Optional parameters
    simplify=0.95,  # Ratio of triangles to remove in the simplification process
    texture_size=1024,  # Size of the texture used for the GLB
)
glb.export(f"{output_dir}/object.glb")

# Save Gaussians as PLY files
outputs["gaussian"][0].save_ply(f"{output_dir}/object.ply")
```

data_process/utils/align_util.py
```python
import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    RasterizationSettings,
    AmbientLights,
    BlendParams,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from scipy.spatial import cKDTree


def select_point(points, raw_matching_points, object_mask):
    mask_coords = np.column_stack(np.where(object_mask > 0))
    mask_coords = mask_coords[:, ::-1]
    tree = cKDTree(mask_coords)

    distances, indices = tree.query(raw_matching_points)

    new_match = mask_coords[indices]
    # Pay attention to the case that the keypoint is in different order
    matched_points = points[new_match[:, 1], new_match[:, 0]]
    return mask_coords[indices], matched_points


def plot_mesh_with_points(mesh, points, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        mesh.vertices[:, 0],
        mesh.vertices[:, 1],
        mesh.vertices[:, 2],
        triangles=mesh.faces,
        alpha=0.5,
        edgecolor="none",
        color="lightgrey",
    )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="red", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    ax.set_title("3D Mesh with Projected Points")
    plt.savefig(filename)
    plt.clf()


def plot_image_with_points(image, points, save_dir, points2=None):
    plt.imshow(image)
    plt.scatter(points[:, 0], points[:, 1], color="red", s=5)
    if points2 is not None:
        plt.scatter(points2[:, 0], points2[:, 1], color="blue", s=5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Points on Original Image")
    plt.savefig(save_dir)
    plt.clf()


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):

        # Extract all meshes from the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        # Combine all meshes if there are multiple
        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")

        # Get model metadata
        metadata = {
            "vertices": combined_mesh.vertices.shape[0],
            "faces": combined_mesh.faces.shape[0],
            "bounds": combined_mesh.bounds.tolist(),
            "center_mass": combined_mesh.center_mass.tolist(),
            "is_watertight": combined_mesh.is_watertight,
            "original_scene": combined_mesh,  # Keep reference to original scene
        }

        mesh = combined_mesh
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def project_2d_to_3d(image_points, depth, camera_intrinsics, camera_pose):
    """
    Project 2D image points to 3D space using the depth map, camera intrinsics, and pose.

    :param image_points: Nx2 array of image points
    :param depth: Depth map
    :param camera_intrinsics: Camera intrinsic matrix
    :param camera_pose: 4x4 camera pose matrix
    :return: Nx3 array of 3D points in world coordinates
    """
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    # Convert image points to normalized device coordinates (NDC)
    ndc_points = np.zeros((image_points.shape[0], 3))
    for i, (u, v) in enumerate(image_points):
        z = depth[int(v), int(u)]
        x = -(u - cx) * z / fx
        y = -(v - cy) * z / fy
        ndc_points[i] = [x, y, z]
    valid_mask = ndc_points[:, 2] > 0
    ndc_points = ndc_points[valid_mask]
    # ndc_points = np.vstack((ndc_points, np.zeros(3), [[0, 0, 0]])) # modified
    # Convert from camera coordinates to world coordinates
    ndc_points_homogeneous = np.hstack((ndc_points, np.ones((ndc_points.shape[0], 1))))
    world_points_homogeneous = ndc_points_homogeneous @ np.linalg.inv(camera_pose)
    return world_points_homogeneous[:, :3], valid_mask


def sample_camera_poses(radius, num_samples, num_up_samples=4, device="cpu"):
    """
    Generate camera poses around a sphere with a given radius.
    camera_poses: A list of 4x4 transformation matrices representing the camera poses.
    camera_view_coord = word_coord @ camera_pose
    """
    camera_poses = []
    phi = np.linspace(0, np.pi, num_samples)  # Elevation angle
    phi = phi[1:-1]  # Exclude poles
    theta = np.linspace(0, 2 * np.pi, num_samples)  # Azimuthal angle

    # Generate different up vectors
    up_vectors = [np.array([0, 0, 1])]  # z-axis is up
    for i in range(1, num_up_samples):
        angle = (i / num_up_samples) * np.pi * 2
        up = np.array([np.sin(angle), 0, np.cos(angle)])  # Rotate around y-axis
        up_vectors.append(up)

    for p in phi:
        for t in theta:
            for up in up_vectors:
                x = radius * np.sin(p) * np.cos(t)
                y = radius * np.sin(p) * np.sin(t)
                z = radius * np.cos(p)
                position = np.array([x, y, z])[None, :]
                lookat = np.array([0, 0, 0])[None, :]
                up = up[None, :]
                R, T = look_at_view_transform(radius, t, p, False, position, lookat, up)
                camera_pose = np.eye(4)
                camera_pose[:3, :3] = R
                camera_pose[3, :3] = T
                camera_poses.append(camera_pose)

    print("total poses", len(camera_poses))
    return torch.tensor(np.array(camera_poses), device=device)


def render_image(mesh, camera_poses, width=640, height=480, fov=1, device="cpu"):
    camera_poses = torch.tensor(camera_poses, device=device)
    if len(camera_poses.shape) == 2:
        camera_poses = camera_poses[None, :]

    from pytorch3d.io import IO
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat

    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    mesh = io.load_mesh(mesh)
    mesh = mesh.to(device)

    R = camera_poses[:, :3, :3]
    T = camera_poses[:, 3, :3]
    num_poses = camera_poses.shape[0]
    cameras = PerspectiveCameras(
        R=R,
        T=T,
        device=device,
        focal_length=torch.ones(num_poses, 1)
        * 0.5
        * width
        / np.tan(fov / 2),  # Calculate focal length from FOV in radians
        principal_point=torch.tensor((width / 2, height / 2))
        .repeat(num_poses)
        .reshape(-1, 2),  # different order from image_size!!
        image_size=torch.tensor((height, width)).repeat(num_poses).reshape(-1, 2),
        in_ndc=False,
    )

    lights = AmbientLights(device=device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0, 0, 0)),
            cameras=cameras,
            lights=lights,
        ),
    )
    extended_mesh = mesh.extend(num_poses).to(device)
    fragments = renderer.rasterizer(extended_mesh)
    depth = fragments.zbuf.squeeze().cpu().numpy()
    rendered_images = renderer(mesh.extend(num_poses))
    color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)

    return color, depth


def render_multi_images(
    mesh,
    width=640,
    height=480,
    fov=1,
    radius=3.0,
    num_samples=6,
    num_ups=2,
    device="cpu",
):
    # Sample camera poses
    camera_poses = sample_camera_poses(radius, num_samples, num_ups, device)

    # Calculate intrinsics
    fx = 0.5 * width / np.tan(fov / 2)
    fy = fx  # * aspect_ratio
    cx, cy = width / 2, height / 2
    camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    num_cameras = camera_poses.shape[0]

    # Render two times to avoid memory overflow
    split = num_cameras // 2
    color1, depth1 = render_image(
        mesh, camera_poses[:split], width, height, fov, device
    )
    color2, depth2 = render_image(
        mesh, camera_poses[split:], width, height, fov, device
    )
    color = np.concatenate([color1, color2], axis=0)
    depth = np.concatenate([depth1, depth2], axis=0)
    return color, depth, camera_poses, camera_intrinsics
```

data_process/utils/visualizer.py
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import imageio
import torch

from matplotlib import cm
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_video_from_path(path):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def draw_line(rgb, coord_y, coord_x, color, linewidth):
    draw = ImageDraw.Draw(rgb)
    draw.line(
        (coord_y[0], coord_y[1], coord_x[0], coord_x[1]),
        fill=tuple(color),
        width=linewidth,
    )
    return rgb


def add_weighted(rgb, alpha, original, beta, gamma):
    return (rgb * alpha + original * beta + gamma).astype("uint8")


class Visualizer:
    def __init__(
        self,
        save_dir: str = "./results",
        grayscale: bool = False,
        pad_value: int = 0,
        fps: int = 10,
        mode: str = "rainbow",  # 'cool', 'optical_flow'
        linewidth: int = 2,
        show_first_frame: int = 10,
        tracks_leave_trace: int = 0,  # -1 for infinite
    ):
        self.mode = mode
        self.save_dir = save_dir
        if mode == "rainbow":
            self.color_map = cm.get_cmap("gist_rainbow")
        elif mode == "cool":
            self.color_map = cm.get_cmap(mode)
        self.show_first_frame = show_first_frame
        self.grayscale = grayscale
        self.tracks_leave_trace = tracks_leave_trace
        self.pad_value = pad_value
        self.linewidth = linewidth
        self.fps = fps

    def visualize(
        self,
        video: torch.Tensor,  # (B,T,C,H,W)
        tracks: torch.Tensor,  # (B,T,N,2)
        visibility: torch.Tensor = None,  # (B, T, N, 1) bool
        gt_tracks: torch.Tensor = None,  # (B,T,N,2)
        segm_mask: torch.Tensor = None,  # (B,1,H,W)
        filename: str = "video",
        writer=None,  # tensorboard Summary Writer, used for visualization during training
        step: int = 0,
        query_frame=0,
        save_video: bool = True,
        compensate_for_camera_motion: bool = False,
        opacity: float = 1.0,
    ):
        if compensate_for_camera_motion:
            assert segm_mask is not None
        if segm_mask is not None:
            coords = tracks[0, query_frame].round().long()
            segm_mask = segm_mask[0, query_frame][coords[:, 1], coords[:, 0]].long()

        video = F.pad(
            video,
            (self.pad_value, self.pad_value, self.pad_value, self.pad_value),
            "constant",
            255,
        )
        color_alpha = int(opacity * 255)
        tracks = tracks + self.pad_value

        if self.grayscale:
            transform = transforms.Grayscale()
            video = transform(video)
            video = video.repeat(1, 1, 3, 1, 1)

        res_video = self.draw_tracks_on_video(
            video=video,
            tracks=tracks,
            visibility=visibility,
            segm_mask=segm_mask,
            gt_tracks=gt_tracks,
            query_frame=query_frame,
            compensate_for_camera_motion=compensate_for_camera_motion,
            color_alpha=color_alpha,
        )
        if save_video:
            self.save_video(res_video, filename=filename, writer=writer, step=step)
        return res_video

    def save_video(self, video, filename, writer=None, step=0):
        if writer is not None:
            writer.add_video(
                filename,
                video.to(torch.uint8),
                global_step=step,
                fps=self.fps,
            )
        else:
            os.makedirs(self.save_dir, exist_ok=True)
            wide_list = list(video.unbind(1))
            wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]

            # Prepare the video file path
            save_path = os.path.join(self.save_dir, f"{filename}.mp4")

            # Create a writer object
            video_writer = imageio.get_writer(save_path, fps=self.fps)

            # Write frames to the video file
            for frame in wide_list[2:-1]:
                video_writer.append_data(frame)

            video_writer.close()

            print(f"Video saved to {save_path}")

    def draw_tracks_on_video(
        self,
        video: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        gt_tracks=None,
        query_frame=0,
        compensate_for_camera_motion=False,
        color_alpha: int = 255,
    ):
        B, T, C, H, W = video.shape
        _, _, N, D = tracks.shape

        assert D == 2
        assert C == 3
        video = video[0].permute(0, 2, 3, 1).byte().detach().cpu().numpy()  # S, H, W, C
        tracks = tracks[0].long().detach().cpu().numpy()  # S, N, 2
        if gt_tracks is not None:
            gt_tracks = gt_tracks[0].detach().cpu().numpy()

        res_video = []

        # process input video
        for rgb in video:
            res_video.append(rgb.copy())
        vector_colors = np.zeros((T, N, 3))

        if self.mode == "optical_flow":
            import flow_vis

            vector_colors = flow_vis.flow_to_color(tracks - tracks[query_frame][None])
        elif segm_mask is None:
            if self.mode == "rainbow":
                y_min, y_max = (
                    tracks[query_frame, :, 1].min(),
                    tracks[query_frame, :, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if isinstance(query_frame, torch.Tensor):
                        query_frame_ = query_frame[n]
                    else:
                        query_frame_ = query_frame
                    color = self.color_map(norm(tracks[query_frame_, n, 1]))
                    color = np.array(color[:3])[None] * 255
                    vector_colors[:, n] = np.repeat(color, T, axis=0)
            else:
                # color changes with time
                for t in range(T):
                    color = np.array(self.color_map(t / T)[:3])[None] * 255
                    vector_colors[t] = np.repeat(color, N, axis=0)
        else:
            if self.mode == "rainbow":
                vector_colors[:, segm_mask <= 0, :] = 255

                y_min, y_max = (
                    tracks[0, segm_mask > 0, 1].min(),
                    tracks[0, segm_mask > 0, 1].max(),
                )
                norm = plt.Normalize(y_min, y_max)
                for n in range(N):
                    if segm_mask[n] > 0:
                        color = self.color_map(norm(tracks[0, n, 1]))
                        color = np.array(color[:3])[None] * 255
                        vector_colors[:, n] = np.repeat(color, T, axis=0)

            else:
                # color changes with segm class
                segm_mask = segm_mask.cpu()
                color = np.zeros((segm_mask.shape[0], 3), dtype=np.float32)
                color[segm_mask > 0] = np.array(self.color_map(1.0)[:3]) * 255.0
                color[segm_mask <= 0] = np.array(self.color_map(0.0)[:3]) * 255.0
                vector_colors = np.repeat(color[None], T, axis=0)

        #  draw tracks
        if self.tracks_leave_trace != 0:
            for t in range(query_frame + 1, T):
                first_ind = (
                    max(0, t - self.tracks_leave_trace)
                    if self.tracks_leave_trace >= 0
                    else 0
                )
                curr_tracks = tracks[first_ind : t + 1]
                curr_colors = vector_colors[first_ind : t + 1]
                if compensate_for_camera_motion:
                    diff = (
                        tracks[first_ind : t + 1, segm_mask <= 0]
                        - tracks[t : t + 1, segm_mask <= 0]
                    ).mean(1)[:, None]

                    curr_tracks = curr_tracks - diff
                    curr_tracks = curr_tracks[:, segm_mask > 0]
                    curr_colors = curr_colors[:, segm_mask > 0]

                res_video[t] = self._draw_pred_tracks(
                    res_video[t],
                    curr_tracks,
                    curr_colors,
                )
                if gt_tracks is not None:
                    res_video[t] = self._draw_gt_tracks(
                        res_video[t], gt_tracks[first_ind : t + 1]
                    )

        #  draw points
        for t in range(T):
            img = Image.fromarray(np.uint8(res_video[t]))
            for i in range(N):
                coord = (tracks[t, i, 0], tracks[t, i, 1])
                visibile = True
                if visibility is not None:
                    visibile = visibility[0, t, i]
                if coord[0] != 0 and coord[1] != 0:
                    if not compensate_for_camera_motion or (
                        compensate_for_camera_motion and segm_mask[i] > 0
                    ):
                        img = draw_circle(
                            img,
                            coord=coord,
                            radius=int(self.linewidth * 2),
                            color=vector_colors[t, i].astype(int),
                            visible=visibile,
                            color_alpha=color_alpha,
                        )
            res_video[t] = np.array(img)

        #  construct the final rgb sequence
        if self.show_first_frame > 0:
            res_video = [res_video[0]] * self.show_first_frame + res_video[1:]
        return torch.from_numpy(np.stack(res_video)).permute(0, 3, 1, 2)[None].byte()

    def _draw_pred_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3
        tracks: np.ndarray,  # T x 2
        vector_colors: np.ndarray,
        alpha: float = 0.5,
    ):
        T, N, _ = tracks.shape
        rgb = Image.fromarray(np.uint8(rgb))
        for s in range(T - 1):
            vector_color = vector_colors[s]
            original = rgb.copy()
            alpha = (s / T) ** 2
            for i in range(N):
                coord_y = (int(tracks[s, i, 0]), int(tracks[s, i, 1]))
                coord_x = (int(tracks[s + 1, i, 0]), int(tracks[s + 1, i, 1]))
                if coord_y[0] != 0 and coord_y[1] != 0:
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        vector_color[i].astype(int),
                        self.linewidth,
                    )
            if self.tracks_leave_trace > 0:
                rgb = Image.fromarray(
                    np.uint8(
                        add_weighted(
                            np.array(rgb), alpha, np.array(original), 1 - alpha, 0
                        )
                    )
                )
        rgb = np.array(rgb)
        return rgb

    def _draw_gt_tracks(
        self,
        rgb: np.ndarray,  # H x W x 3,
        gt_tracks: np.ndarray,  # T x 2
    ):
        T, N, _ = gt_tracks.shape
        color = np.array((211, 0, 0))
        rgb = Image.fromarray(np.uint8(rgb))
        for t in range(T):
            for i in range(N):
                gt_tracks = gt_tracks[t][i]
                #  draw a red cross
                if gt_tracks[0] > 0 and gt_tracks[1] > 0:
                    length = self.linewidth * 3
                    coord_y = (int(gt_tracks[0]) + length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) - length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
                    coord_y = (int(gt_tracks[0]) - length, int(gt_tracks[1]) + length)
                    coord_x = (int(gt_tracks[0]) + length, int(gt_tracks[1]) - length)
                    rgb = draw_line(
                        rgb,
                        coord_y,
                        coord_x,
                        color,
                        self.linewidth,
                    )
        rgb = np.array(rgb)
        return rgb
```

evaluate_chamfer.py
```python
import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
from pytorch3d.loss import chamfer_distance

prediction_dir = "./experiments"
base_path = "./data/different_types"
output_file = "results/final_results.csv"

if not os.path.exists("results"):
    os.makedirs("results")

def evaluate_prediction(
    start_frame: int,
    end_frame: int,
    vertices: torch.Tensor | np.ndarray,
    object_points: torch.Tensor | np.ndarray,
    object_visibilities: torch.Tensor | np.ndarray,
    object_motions_valid: torch.Tensor | np.ndarray,
    num_original_points: int,
    num_surface_points: int,
) -> dict[str, int | float]:
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


if __name__ == "__main__":
    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)

    writer.writerow(
        [
            "Case Name",
            "Train Frame Num",
            "Train Chamfer Error",
            "Test Frame Num",
            "Test Chamfer Error",
        ]
    )

    dir_names = glob.glob(f"{prediction_dir}/*")
    for dir_name in dir_names:
        case_name = dir_name.split("/")[-1]
        print(f"Processing {case_name}")

        # Read the trajectory data
        with open(f"{dir_name}/inference.pkl", "rb") as f:
            vertices = pickle.load(f)

        # Read the GT object points and masks
        with open(f"{base_path}/{case_name}/final_data.pkl", "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        num_original_points = object_points.shape[1]
        num_surface_points = num_original_points + data["surface_points"].shape[0]

        # read the train/test split
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]
        test_frame = split["test"][1]

        assert (
            test_frame == vertices.shape[0]
        ), f"Test frame {test_frame} != {vertices.shape[0]}"

        # Do the statistics on train split, only evalaute from the 2nd frame
        results_train = evaluate_prediction(
            1,
            train_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )
        results_test = evaluate_prediction(
            train_frame,
            test_frame,
            vertices,
            object_points,
            object_visibilities,
            object_motions_valid,
            num_original_points,
            num_surface_points,
        )

        writer.writerow(
            [
                case_name,
                results_train["frame_len"],
                results_train["chamfer_error"],
                results_test["frame_len"],
                results_test["chamfer_error"],
            ]
        )
    file.close()
```

evaluate_track.py
```python
import pickle
import glob
import csv
import json
import numpy as np
from scipy.spatial import KDTree

base_path = "./data/different_types"
prediction_path = "experiments"
output_file = "results/final_track.csv"


def evaluate_prediction(
    start_frame: int,
    end_frame: int,
    vertices: np.ndarray,
    gt_track_3d: np.ndarray,
    idx: np.ndarray,
    mask: np.ndarray,
) -> float:
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        
        track_errors.append(track_error)
    return np.mean(track_errors)


file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = csv.writer(file)
writer.writerow(
    [
        "Case Name",
        "Train Track Error",
        "Test Track Error",
    ]
)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    # if case_name != "single_lift_dinosor":
    #     continue
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    with open(f"{prediction_path}/{case_name}/inference.pkl", "rb") as f:
        vertices = pickle.load(f)

    with open(f"{base_path}/{case_name}/gt_track_3d.pkl", "rb") as f:
        gt_track_3d = pickle.load(f)

    # Locate the index of corresponding point index in the vertices, if nan, then ignore the points
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)

    kdtree = KDTree(vertices[0])
    dis, idx = kdtree.query(gt_track_3d[0][mask])

    train_track_error = evaluate_prediction(
        1, train_frame, vertices, gt_track_3d, idx, mask
    )
    test_track_error = evaluate_prediction(
        train_frame, test_frame, vertices, gt_track_3d, idx, mask
    )
    writer.writerow([case_name, train_track_error, test_track_error])
file.close()
```

export_gaussian_data.py
```python
import os
import csv
import json
import pickle
import numpy as np
import open3d as o3d

base_path = "./data/different_types"
output_path = "./data/gaussian_data"
CONTROLLER_NAME = "hand"


def existDir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        print(f"Processing {case_name}!!!!!!!!!!!!!!!")

        # Create the directory for the case
        existDir(f"{output_path}/{case_name}")
        for i in range(3):
            # Copy the original RGB image
            os.system(
                f"cp {base_path}/{case_name}/color/{i}/0.png {output_path}/{case_name}/{i}.png"
            )
            # Copy the original mask image
            # Get the mask path for the image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            mask_path = f"{base_path}/{case_name}/mask/{i}/{obj_idx}/0.png"
            os.system(f"cp {mask_path} {output_path}/{case_name}/mask_{i}.png")
            # Prepare the high-resolution image
            os.system(
                f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/{i}/0.png --output_path {output_path}/{case_name}/{i}_high.png --category {category}"
            )
            # Prepare the segmentation mask of the high-resolution image
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT {category} --output_path {output_path}/{case_name}/mask_{i}_high.png"
            )

            # Copy the original depth image
            os.system(
                f"cp {base_path}/{case_name}/depth/{i}/0.npy {output_path}/{case_name}/{i}_depth.npy"
            )

            # Prepare the human mask for the low-resolution image and high-resolution image
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}.png"
            )
            os.system(
                f"python ./data_process/segment_util_image.py --img_path {output_path}/{case_name}/{i}_high.png --TEXT_PROMPT 'human' --output_path {output_path}/{case_name}/mask_human_{i}_high.png"
            )

        # Prepare the intrinsic and extrinsic parameters
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            intrinsics = json.load(f)["intrinsics"]
        data = {}
        data["c2ws"] = c2ws
        data["intrinsics"] = intrinsics
        with open(f"{output_path}/{case_name}/camera_meta.pkl", "wb") as f:
            pickle.dump(data, f)

        # Prepare the shape initialization data
        # If with shape prior, then copy the shape prior data
        if shape_prior.lower() == "true":
            os.system(
                f"cp {base_path}/{case_name}/shape/matching/final_mesh.glb {output_path}/{case_name}/shape_prior.glb"
            )
        # Save the original pcd data into the world coordinate system
        obs_points = []
        obs_colors = []
        pcd_path = f"{base_path}/{case_name}/pcd/0.npz"
        processed_mask_path = f"{base_path}/{case_name}/mask/processed_masks.pkl"
        data = np.load(pcd_path)
        with open(processed_mask_path, "rb") as f:
            processed_masks = pickle.load(f)
        for i in range(3):
            points = data["points"][i]
            colors = data["colors"][i]
            mask = processed_masks[0][i]["object"]
            obs_points.append(points[mask])
            obs_colors.append(colors[mask])

        obs_points = np.vstack(obs_points)
        obs_colors = np.vstack(obs_colors)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obs_points)
        pcd.colors = o3d.utility.Vector3dVector(obs_colors)
        # coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        # o3d.visualization.draw_geometries([pcd, coordinate])
        o3d.io.write_point_cloud(f"{output_path}/{case_name}/observation.ply", pcd)
```

export_render_eval_data.py
```python
import os
import csv
import json

base_path = "./data/different_types"
output_path = "./data/render_eval_data"
CONTROLLER_NAME = "hand"


def existDir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


existDir(output_path)

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]
        
        if not os.path.exists(f"{base_path}/{case_name}"):
            continue
        print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    
        # Create the directory for the case
        existDir(f"{output_path}/{case_name}")
        existDir(f"{output_path}/{case_name}/mask")
        for i in range(3):
            # Copy the original RGB image
            os.system(
                f"cp -r {base_path}/{case_name}/color {output_path}/{case_name}/"
            )
            # Copy only the object mask image
            # Get the mask path for the image
            with open(f"{base_path}/{case_name}/mask/mask_info_{i}.json", "r") as f:
                data = json.load(f)
            obj_idx = None
            for key, value in data.items():
                if value != CONTROLLER_NAME:
                    if obj_idx is not None:
                        raise ValueError("More than one object detected.")
                    obj_idx = int(key)
            existDir(f"{output_path}/{case_name}/mask/{i}")
            os.system(f"cp -r {base_path}/{case_name}/mask/{i}/{obj_idx}/* {output_path}/{case_name}/mask/{i}/")
        
        # Copy the split.json
        os.system(f"cp {base_path}/{case_name}/split.json {output_path}/{case_name}/")```

export_video_human_mask.py
```python
import os
import glob

base_path = "./data/different_types"
output_path = "./data/different_types_human_mask"

def existDir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")
    existDir(f"{output_path}/{case_name}")
    # Process to get the whole human mask for the video

    TEXT_PROMPT = "human"
    camera_num = 3
    assert len(glob.glob(f"{base_path}/{case_name}/depth/*")) == camera_num

    for camera_idx in range(camera_num):
        print(f"Processing {case_name} camera {camera_idx}")
        os.system(
            f"python ./data_process/segment_util_video.py --output_path {output_path}/{case_name} --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT} --camera_idx {camera_idx}"
        )
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")```

gaussian_splatting/__init__.py
```python
```

gaussian_splatting/arguments/__init__.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = 1         # avoid downsampling
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.gs_init_opt = "pcd"     # 'pcd', 'mesh', 'hybrid'
        self.pts_per_triangles = 30  # number of points per triangle sampled from the mesh to init gs
        self.use_high_res = False    # set to True if want to use 4x up-sampled images.
        self.use_masks = False       # set to True if want to use foreground object masks.
        self.isotropic = False       # set to True if want to use isotropic splatting.
        self.disable_sh = False      # set to True if want to disable SH coefficients during rendering.
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.lambda_depth = 0.0        # 1e-1
        self.lambda_normal = 0.0       # 1e-3
        self.lambda_anisotropic = 0.0  # 1e-1
        self.lambda_seg = 0.0          # 1e-1
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
```

gaussian_splatting/dynamic_utils.py
```python
from __future__ import annotations

import torch
import kornia


from torch import Tensor


def quat2mat(q: Tensor) -> Tensor:
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.shape[0], 3, 3)).to(q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def mat2quat(rot: Tensor) -> Tensor:
    t = torch.clamp(rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=-1)
    q = torch.zeros((rot.shape[0], 4)).to(rot.device)

    mask_0 = t > -1
    t_0 = torch.sqrt(t[mask_0] + 1)
    q[mask_0, 0] = 0.5 * t_0
    t_0 = 0.5 / t_0
    q[mask_0, 1] = (rot[mask_0, 2, 1] - rot[mask_0, 1, 2]) * t_0
    q[mask_0, 2] = (rot[mask_0, 0, 2] - rot[mask_0, 2, 0]) * t_0
    q[mask_0, 3] = (rot[mask_0, 1, 0] - rot[mask_0, 0, 1]) * t_0

    # i = 0, j = 1, k = 2
    mask_1 = ~mask_0 & (rot[:, 0, 0] >= rot[:, 1, 1]) & (rot[:, 0, 0] >= rot[:, 2, 2])
    t_1 = torch.sqrt(1 + rot[mask_1, 0, 0] - rot[mask_1, 1, 1] - rot[mask_1, 2, 2])
    t_1 = 0.5 / t_1
    q[mask_1, 0] = (rot[mask_1, 2, 1] - rot[mask_1, 1, 2]) * t_1
    q[mask_1, 1] = 0.5 * t_1
    q[mask_1, 2] = (rot[mask_1, 1, 0] + rot[mask_1, 0, 1]) * t_1
    q[mask_1, 3] = (rot[mask_1, 2, 0] + rot[mask_1, 0, 2]) * t_1

    # i = 1, j = 2, k = 0
    mask_2 = ~mask_0 & (rot[:, 1, 1] >= rot[:, 2, 2]) & (rot[:, 1, 1] > rot[:, 0, 0])
    t_2 = torch.sqrt(1 + rot[mask_2, 1, 1] - rot[mask_2, 0, 0] - rot[mask_2, 2, 2])
    t_2 = 0.5 / t_2
    q[mask_2, 0] = (rot[mask_2, 0, 2] - rot[mask_2, 2, 0]) * t_2
    q[mask_2, 1] = (rot[mask_2, 2, 1] + rot[mask_2, 1, 2]) * t_2
    q[mask_2, 2] = 0.5 * t_2
    q[mask_2, 3] = (rot[mask_2, 0, 1] + rot[mask_2, 1, 0]) * t_2

    # i = 2, j = 0, k = 1
    mask_3 = ~mask_0 & (rot[:, 2, 2] > rot[:, 0, 0]) & (rot[:, 2, 2] > rot[:, 1, 1])
    t_3 = torch.sqrt(1 + rot[mask_3, 2, 2] - rot[mask_3, 0, 0] - rot[mask_3, 1, 1])
    t_3 = 0.5 / t_3
    q[mask_3, 0] = (rot[mask_3, 1, 0] - rot[mask_3, 0, 1]) * t_3
    q[mask_3, 1] = (rot[mask_3, 0, 2] + rot[mask_3, 2, 0]) * t_3
    q[mask_3, 2] = (rot[mask_3, 1, 2] + rot[mask_3, 2, 1]) * t_3
    q[mask_3, 3] = 0.5 * t_3

    assert torch.allclose(mask_1 + mask_2 + mask_3 + mask_0, torch.ones_like(mask_0))
    return q

def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, k)
    # indices: (n_bones,)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, n_bones)

    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        weights = torch.ones((n_particles, n_bones), device=device)

        dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
        dist = torch.clamp(dist, min=1e-4)
        weights = 1 / dist
        # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
        # weights[weights < weights_topk.values[:, -1:]] = 0.
        weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
        # weights[weights < 0.01] = 0.
        # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights


def create_relation_matrix(points, K=5):
    """
    Create an NxN relation matrix where each row has 1s for the top K closest neighbors and 0s elsewhere.
    
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to mark as 1.
        
    Returns:
        torch.Tensor: NxN relation matrix with dtype int.
    """
    N = points.shape[0]

    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    # Create the NxN relation matrix
    relation_matrix = torch.zeros((N, N), dtype=torch.int)

    # Scatter 1s for the top K neighbors
    batch_indices = torch.arange(N).unsqueeze(1).expand(-1, K)
    relation_matrix[batch_indices, topk_indices] = 1

    return relation_matrix


def get_topk_indices(points, K=5):
    """
    Compute the indices of the top K closest neighbors for each point.

    Args:
        points (torch.Tensor): Tensor of shape (N, 3) representing 3D points.
        K (int): Number of closest neighbors to retrieve.

    Returns:
        torch.Tensor: Tensor of shape (N, K) containing the indices of the top K closest neighbors.
    """
    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # (N, N)

    # Get the indices of the top K closest neighbors (excluding self)
    topk_indices = torch.topk(dist_matrix, K + 1, largest=False).indices[:, 1:]  # Skip self (0 distance)

    return topk_indices


def knn_weights(bones, pts, K=5):
    dist = torch.norm(pts[:, None] - bones, dim=-1)  # (n_pts, n_bones)
    _, indices = torch.topk(dist, K, dim=-1, largest=False)
    bones_selected = bones[indices]  # (N, k, 3)
    dist = torch.norm(bones_selected - pts[:, None], dim=-1)  # (N, k)
    weights = 1 / (dist + 1e-6)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # (N, k)
    weights_all = torch.zeros((pts.shape[0], bones.shape[0]), device=pts.device)  # TODO: prevent init new one
    # weights_all[torch.arange(pts.shape[0])[:, None], indices] = weights
    weights_all[torch.arange(pts.shape[0], device=pts.device)[:, None], indices] = weights
    return weights_all



def calc_weights_vals_from_indices(bones, pts, indices):
    # bones: (n_bones, 3)
    # pts: (n_particles, 3)
    # indices: (n_particles, k) indices of k nearest bones per particle

    nearest_bones = bones[indices]  # (n_particles, k, 3)
    pts_expanded = pts.unsqueeze(1)  # (n_particles, 1, 3)
    distances = torch.norm(pts_expanded - nearest_bones, dim=2)
    weights_vals = 1.0 / (distances + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=1, keepdim=True)  # (n_particles, k)    
    return weights_vals


def knn_weights_sparse(bones, pts, K=5):
    dist = torch.norm(pts[:, None].cpu() - bones.cpu(), dim=-1)  # (n_pts, n_bones)
    weights_vals, indices = torch.topk(dist, K, dim=-1, largest=False)
    weights_vals = weights_vals.to(pts.device)
    indices = indices.to(pts.device)
    weights_vals = 1 / (weights_vals + 1e-6)
    weights_vals = weights_vals / weights_vals.sum(dim=-1, keepdim=True)  # (N, k)
    torch.cuda.empty_cache()
    return weights_vals, indices

def interpolate_motions_speedup(bones, motions, relations, xyz, rot=None, quat=None, weights=None, weights_indices=None, device='cuda', step='n/a'):
    # bones: (n_bones, 3) bone positions
    # motions: (n_bones, 3) bone motions/displacements
    # relations: (n_bones, k_adj) bone adjacency relationships - which bones are connected to each other
    # xyz: (n_particles, 3) particle positions
    # weights: (n_particles, k) weights for k nearest bones per particle
    # weights_indices: (n_particles, k) indices of k nearest bones per particle
    # rot: (n_particles, 3, 3) optional rotation matrices
    # quat: (n_particles, 4) optional quaternions

    n_bones, _ = bones.shape
    n_particles, k_nearest = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros((n_bones, 4, 4),  device=device)

    n_adj = relations.shape[1]
    
    adj_bones = bones[relations] - bones[:, None]  # (n_bones, n_adj, 3)
    adj_bones_new = (bones[relations] + motions[relations]) - (bones[:, None] + motions[:, None])  # (n_bones, n_adj, 3)

    W = torch.eye(n_adj, device=device)[None].repeat(n_bones, 1, 1)  # (n_bones, n_adj, n_adj)

    # fit a transformation
    F = adj_bones_new.permute(0, 2, 1) @ W @ adj_bones  # (n_bones, 3, 3)
    
    cov_rank = torch.linalg.matrix_rank(F)  # (n_bones,)
    
    cov_rank_3_mask = cov_rank == 3  # (n_bones,)
    cov_rank_2_mask = cov_rank == 2  # (n_bones,)
    cov_rank_1_mask = cov_rank == 1  # (n_bones,)

    F_2_3 = F[cov_rank_2_mask | cov_rank_3_mask]  # (n_bones, 3, 3)
    F_1 = F[cov_rank_1_mask]  # (n_bones, 3, 3)

    # 2 or 3
    try:
        U, S, V = torch.svd(F_2_3)  # S: (n_bones, 3)
        S = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(F_2_3.shape[0], 1, 1)
        neg_det_mask = torch.linalg.det(F_2_3) < 0
        if neg_det_mask.sum() > 0:
            print(f'[step {step}] F det < 0 for {neg_det_mask.sum()} bones')
            S[neg_det_mask, -1, -1] = -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)
    except:
        print(f'[step {step}] SVD failed')
        import ipdb; ipdb.set_trace()

    neg_1_det_mask = torch.abs(torch.linalg.det(R) + 1) < 1e-3
    pos_1_det_mask = torch.abs(torch.linalg.det(R) - 1) < 1e-3
    bad_det_mask = ~(neg_1_det_mask | pos_1_det_mask)

    if neg_1_det_mask.sum() > 0:
        print(f'[step {step}] det -1')
        S[neg_1_det_mask, -1, -1] *= -1  # S[:, -1, -1] or S[:, cov_rank, cov_rank] or S[:, cov_rank - 1, cov_rank - 1]?
        R = U @ S @ V.permute(0, 2, 1)

    try:
        assert bad_det_mask.sum() == 0
    except:
        print(f'[step {step}] Bad det')
        import ipdb; ipdb.set_trace()

    try:
        if cov_rank_1_mask.sum() > 0:
            print(f'[step {step}] F rank 1 for {cov_rank_1_mask.sum()} bones')
            U, S, V = torch.svd(F_1)  # S: (n_bones', 3)
            assert torch.allclose(S[:, 1:], torch.zeros_like(S[:, 1:]))
            x = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)[None].repeat(F_1.shape[0], 1)  # (n_bones', 3)
            axis = U[:, :, 0]  # (n_bones', 3)
            perp_axis = torch.linalg.cross(axis, x)  # (n_bones', 3)

            perp_axis_norm_mask = torch.norm(perp_axis, dim=1) < 1e-6

            R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
            if perp_axis_norm_mask.sum() > 0:
                print(f'[step {step}] Perp axis norm 0 for {perp_axis_norm_mask.sum()} bones')
                R[perp_axis_norm_mask] = torch.eye(3, device=device, dtype=torch.float32)[None].repeat(perp_axis_norm_mask.sum(), 1, 1)

            perp_axis = perp_axis[~perp_axis_norm_mask]  # (n_bones', 3)
            x = x[~perp_axis_norm_mask]  # (n_bones', 3)

            perp_axis = perp_axis / torch.norm(perp_axis, dim=1, keepdim=True)  # (n_bones', 3)
            third_axis = torch.linalg.cross(x, perp_axis)  # (n_bones', 3)
            assert ((torch.norm(third_axis, dim=1) - 1).abs() < 1e-6).all()
            third_axis_after = torch.linalg.cross(axis, perp_axis)  # (n_bones', 3)

            X = torch.stack([x, perp_axis, third_axis], dim=-1)
            Y = torch.stack([axis, perp_axis, third_axis_after], dim=-1)
            R[~perp_axis_norm_mask] = Y @ X.permute(0, 2, 1)
    except:
        R = torch.zeros((F_1.shape[0], 3, 3), device=device, dtype=torch.float32)
        R[:, 0, 0] = 1
        R[:, 1, 1] = 1
        R[:, 2, 2] = 1

    try:
        bone_transforms[:, :3, :3] = R
    except:
        print(f'[step {step}] Bad R')
        bone_transforms[:, 0, 0] = 1
        bone_transforms[:, 1, 1] = 1
        bone_transforms[:, 2, 2] = 1
    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    # if weights is None:
    #     weights = torch.ones((n_particles, n_bones), device=device)

    #     dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
    #     dist = torch.clamp(dist, min=1e-4)
    #     weights = 1 / dist
    #     # weights_topk = torch.topk(weights, 5, dim=1, largest=True, sorted=True)
    #     # weights[weights < weights_topk.values[:, -1:]] = 0.
    #     weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    #     # weights[weights < 0.01] = 0.
    #     # weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    # xyz_transformed = torch.zeros((n_particles, n_bones, 3), device=device)

    # xyz_transformed = xyz[:, None] - bones[None]  # (n_particles, n_bones, 3)
    # # xyz_transformed = (bone_transforms[:, :3, :3][None].repeat(n_particles, 1, 1, 1)\
    # #         .reshape(n_particles * n_bones, 3, 3) @ xyz_transformed.reshape(n_particles * n_bones, 3, 1)).reshape(n_particles, n_bones, 3)
    # xyz_transformed = torch.einsum('ijk,jkl->ijl', xyz_transformed, bone_transforms[:, :3, :3].permute(0, 2, 1))  # (n_particles, n_bones, 3)
    # xyz_transformed = xyz_transformed + bone_transforms[:, :3, 3][None] + bones[None]  # (n_particles, n_bones, 3)
    # xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)


    selected_bones = bones[weights_indices]  # (n_particles, k, 3)
    selected_transforms = bone_transforms[weights_indices]  # (n_particles, k, 4, 4)

    # Transform each point with only its k nearest bones
    # xyz_expanded = xyz[:, None].unsqueeze(1).expand(-1, k_nearest, -1)  # (n_particles, k, 3)
    # xyz_local = xyz_expanded - selected_bones  # (n_particles, k, 3)
    xyz_local = xyz.unsqueeze(1) - selected_bones  # (n_particles, k, 3)
    
    # Apply rotation to local coordinates 
    rotated_local = torch.einsum('nkij,nkj->nki', selected_transforms[:, :, :3, :3], xyz_local)  # (n_particles, k, 3)
    
    # Apply translation and add back bone positions
    transformed_pts = rotated_local + selected_transforms[:, :, :3, 3] + selected_bones  # (n_particles, k, 3)
    
    # Apply weights to get final positions
    xyz_transformed = torch.sum(transformed_pts * weights[:, :, None], dim=1)  # (n_particles, 3)


    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros_like(q1)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        # base_quats = kornia.geometry.conversions.rotation_matrix_to_quaternion(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        # base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        # quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        # quats = torch.nn.functional.normalize(quats, dim=-1)

        from kornia.geometry.conversions import rotation_matrix_to_quaternion

        selected_rot_matrices = selected_transforms[:, :, :3, :3]  # (n_particles, k, 3, 3)
        n_particles, k_weights = weights_indices.shape
        batch_rot_matrices = selected_rot_matrices.reshape(-1, 3, 3)  # (n_particles*k, 3, 3)
        
        try:
            base_quats = rotation_matrix_to_quaternion(batch_rot_matrices)  # (n_particles*k, 4)
        except:
            print('use mat2quat')
            base_quats = mat2quat(batch_rot_matrices)  # (n_particles*k, 4)
            
        base_quats = base_quats.reshape(n_particles, k_weights, 4)  # (n_particles, k, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)
        quats = torch.sum(base_quats * weights[:, :, None], dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)

        rot = quaternion_multiply(quats, quat)

    # Return sparse weights representation for reuse
    weights_sparse = (weights, weights_indices)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights_sparse```

gaussian_splatting/evaluate_render.py
```python
import os
from PIL import Image
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr
import json
from tqdm import tqdm
import torch
# import torchvision.transforms.functional as tf
import torchvision.transforms as transforms
import numpy as np


def img2tensor(img):
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = img.transpose(2, 0, 1)  # Change shape from (H, W, C) to (C, H, W)
    return torch.from_numpy(img).unsqueeze(0).cuda()


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 1.0


if __name__ == "__main__":
    render_path = './data/render_eval_data'
    human_mask_path = "./data/different_types_human_mask"
    root_data_dir = './data/gaussian_data'
    output_dir = './gaussian_output_dynamic'

    log_dir = './results'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'output_dynamic.txt')

    with open(log_file_path, 'w') as log_file:

        scene_name = sorted(os.listdir(render_path))

        all_psnrs_train, all_ssims_train, all_lpipss_train, all_ious_train = [], [], [], []
        all_psnrs_test, all_ssims_test, all_lpipss_test, all_ious_test = [], [], [], []

        scene_metrics = {}

        for scene in scene_name:

            scene_dir = os.path.join(root_data_dir, scene)
            output_scene_dir = os.path.join(output_dir, scene)
            render_path_dir = os.path.join(render_path, scene)
            human_mask_dir = os.path.join(human_mask_path, scene)

            # Load frame split info
            with open(f"{render_path_dir}/split.json", 'r') as f:
                info = json.load(f)
            frame_len = info['frame_len']
            train_f_idx_range = list(range(info["train"][0] + 1, info["train"][1]))   # +1 if ignoring the first frame
            test_f_idx_range = list(range(info["test"][0], info["test"][1]))

            print("train indices range from", train_f_idx_range[0], "to", train_f_idx_range[-1])
            print("test indices range from", test_f_idx_range[0], "to", test_f_idx_range[-1])

            psnrs_train, ssims_train, lpipss_train, ious_train = [], [], [], []
            psnrs_test, ssims_test, lpipss_test, ious_test = [], [], [], []

            # for view_idx in range(3):
            for view_idx in range(1):   # only consider the first view

                for frame_idx in train_f_idx_range:
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_train.append(psnr(render_tensor, gt_tensor).item())
                    ssims_train.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_train.append(lpips(render_tensor, gt_tensor).item())
                    ious_train.append(compute_iou(gt_mask > 0, render_mask > 0))

                for frame_idx in test_f_idx_range:
                        
                    gt = np.array(Image.open(os.path.join(render_path_dir, 'color', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = np.array(Image.open(os.path.join(render_path_dir, 'mask', str(view_idx), f'{frame_idx}.png')))
                    gt_mask = gt_mask.astype(np.float32) / 255.

                    render = np.array(Image.open(os.path.join(output_scene_dir, str(view_idx), f'{frame_idx:05d}.png')))
                    render_mask = render[:, :, 3] if render.shape[-1] == 4 else np.ones_like(render[:, :, 0])

                    human_mask = np.array(Image.open(os.path.join(human_mask_dir, 'mask', str(view_idx), '0', f'{frame_idx}.png')))
                    inv_human_mask = (1.0 - human_mask / 255.).astype(np.float32)

                    gt = gt.astype(np.float32) * gt_mask[..., None]
                    bg_mask = gt_mask == 0
                    gt[bg_mask] = [0, 0, 0]
                    render = render[:, :, :3].astype(np.float32)

                    gt = gt * inv_human_mask[..., None]
                    render = render * inv_human_mask[..., None]
                    render_mask = render_mask * inv_human_mask

                    gt_tensor = img2tensor(gt)
                    render_tensor = img2tensor(render)

                    psnrs_test.append(psnr(render_tensor, gt_tensor).item())
                    ssims_test.append(ssim(render_tensor, gt_tensor).item())
                    lpipss_test.append(lpips(render_tensor, gt_tensor).item())
                    ious_test.append(compute_iou(gt_mask > 0, render_mask > 0))

            scene_metrics[scene] = {
                'psnr_train': np.mean(psnrs_train),
                'ssim_train': np.mean(ssims_train),
                'lpips_train': np.mean(lpipss_train),
                'iou_train': np.mean(ious_train),
                'psnr_test': np.mean(psnrs_test),
                'ssim_test': np.mean(ssims_test),
                'lpips_test': np.mean(lpipss_test),
                'iou_test': np.mean(ious_test)
            }

            all_psnrs_train.extend(psnrs_train)
            all_ssims_train.extend(ssims_train)
            all_lpipss_train.extend(lpipss_train)
            all_ious_train.extend(ious_train)

            all_psnrs_test.extend(psnrs_test)
            all_ssims_test.extend(ssims_test)
            all_lpipss_test.extend(lpipss_test)
            all_ious_test.extend(ious_test)

            print(f'===== Scene: {scene} =====')
            print(f'\t PSNR (train): {np.mean(psnrs_train):.4f}')
            print(f'\t SSIM (train): {np.mean(ssims_train):.4f}')
            print(f'\t LPIPS (train): {np.mean(lpipss_train):.4f}')
            print(f'\t IoU (train): {np.mean(ious_train):.4f}')

            print(f'\t PSNR (test): {np.mean(psnrs_test):.4f}')
            print(f'\t SSIM (test): {np.mean(ssims_test):.4f}')
            print(f'\t LPIPS (test): {np.mean(lpipss_test):.4f}')
            print(f'\t IoU (test): {np.mean(ious_test):.4f}')

        print('===== Overall Results Across All Scenes =====')
        print(f'\t Overall PSNR (train): {np.mean(all_psnrs_train):.4f}')
        print(f'\t Overall SSIM (train): {np.mean(all_ssims_train):.4f}')
        print(f'\t Overall LPIPS (train): {np.mean(all_lpipss_train):.4f}')
        print(f'\t Overall IoU (train): {np.mean(all_ious_train):.4f}')

        print(f'\t Overall PSNR (test): {np.mean(all_psnrs_test):.4f}')
        print(f'\t Overall SSIM (test): {np.mean(all_ssims_test):.4f}')
        print(f'\t Overall LPIPS (test): {np.mean(all_lpipss_test):.4f}')
        print(f'\t Overall IoU (test): {np.mean(all_ious_test):.4f}')

        overall_psnr_train = np.mean(all_psnrs_train)
        overall_ssim_train = np.mean(all_ssims_train)
        overall_lpips_train = np.mean(all_lpipss_train)
        overall_iou_train = np.mean(all_ious_train)
        
        overall_psnr_test = np.mean(all_psnrs_test)
        overall_ssim_test = np.mean(all_ssims_test)
        overall_lpips_test = np.mean(all_lpipss_test)
        overall_iou_test = np.mean(all_ious_test)

        # Write overall metrics to log file
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("OVERALL RESULTS ACROSS ALL SCENES\n")
        log_file.write("=" * 80 + "\n\n")
        
        log_file.write(f"Overall PSNR (train): {overall_psnr_train:.6f}\n")
        log_file.write(f"Overall SSIM (train): {overall_ssim_train:.6f}\n")
        log_file.write(f"Overall LPIPS (train): {overall_lpips_train:.6f}\n")
        log_file.write(f"Overall IoU (train): {overall_iou_train:.6f}\n\n")
        
        log_file.write(f"Overall PSNR (test): {overall_psnr_test:.6f}\n")
        log_file.write(f"Overall SSIM (test): {overall_ssim_test:.6f}\n")
        log_file.write(f"Overall LPIPS (test): {overall_lpips_test:.6f}\n")
        log_file.write(f"Overall IoU (test): {overall_iou_test:.6f}\n\n")
        
        # Create a compact table of all scene metrics
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("COMPACT METRICS TABLE BY SCENE\n")
        log_file.write("=" * 80 + "\n\n")
        
        # Header
        log_file.write(f"{'Scene':<50} | {'PSNR-train':<12} | {'SSIM-train':<12} | {'LPIPS-train':<14} | {'IoU-train':<12} | ")
        log_file.write(f"{'PSNR-test':<12} | {'SSIM-test':<12} | {'LPIPS-test':<14} | {'IoU-test':<12}\n")
        log_file.write("-" * 160 + "\n")
        
        # Scene rows
        for scene in scene_name:
            metrics = scene_metrics[scene]
            log_file.write(f"{scene[:50]:<50} | ")
            log_file.write(f"{metrics['psnr_train']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_train']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_train']:<14.6f} | ")
            log_file.write(f"{metrics['iou_train']:<12.6f} | ")
            
            log_file.write(f"{metrics['psnr_test']:<12.6f} | ")
            log_file.write(f"{metrics['ssim_test']:<12.6f} | ")
            log_file.write(f"{metrics['lpips_test']:<14.6f} | ")
            log_file.write(f"{metrics['iou_test']:<12.6f}\n")
        
        # Overall row
        log_file.write("-" * 160 + "\n")
        log_file.write(f"{'OVERALL':<50} | ")
        log_file.write(f"{overall_psnr_train:<12.6f} | ")
        log_file.write(f"{overall_ssim_train:<12.6f} | ")
        log_file.write(f"{overall_lpips_train:<14.6f} | ")
        log_file.write(f"{overall_iou_train:<12.6f} | ")
        
        log_file.write(f"{overall_psnr_test:<12.6f} | ")
        log_file.write(f"{overall_ssim_test:<12.6f} | ")
        log_file.write(f"{overall_lpips_test:<14.6f} | ")
        log_file.write(f"{overall_iou_test:<12.6f}\n")
        
        print(f"\nMetrics have been saved to: {log_file_path}")```

gaussian_splatting/gaussian_renderer/__init__.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from ..scene.gaussian_model import GaussianModel
from ..utils.sh_utils import eval_sh
from torch.nn import functional as F
from gsplat import rasterization


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, use_gsplat=True, antialiased=False, separate_sh = False, use_trained_exp=False):
    if use_gsplat:
        return render_gsplat(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color, antialiased)
    else:
        return render_3dgs(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, separate_sh, override_color, use_trained_exp)


# This is code is adapted from ChatSim background gaussians model: 
# https://github.com/yifanlu0227/ChatSim/blob/main/chatsim/background/gaussian-splatting/gaussian_renderer/gsplat_renderer.py
def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, antialiased = True, render_normals = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    if viewpoint_camera.K is not None:
        # print("====== Use camera K ======")
        # focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K
        focal_length_x, focal_length_y, cx, cy = viewpoint_camera.K[0, 0], viewpoint_camera.K[1, 1], viewpoint_camera.K[0, 2], viewpoint_camera.K[1, 2]
        K = torch.tensor([
            [focal_length_x, 0, cx],
            [0, focal_length_y, cy],
            [0, 0, 1.0]
        ]).to(pc.get_xyz)
    else:
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
        focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
        K = torch.tensor(
            [
                [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
                [0, focal_length_y, viewpoint_camera.image_height / 2.0],
                [0, 0, 1],
            ]
        ).to(pc.get_xyz)

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1) # [4, 4]

    rasterize_mode = 'classic' if not antialiased else 'antialiased'

    render_colors, render_alphas, info = rasterization(
        means=means3D,    # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,    # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+ED',
        rasterize_mode=rasterize_mode,
        absgrad=True
    )
    # [1, H, W, 4] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)[:3]
    # [1, H, W, 4] -> [1, H, W]
    rendered_depth = render_colors[0].permute(2, 0, 1)[3:]
    # [1, H, W, 1] -> [1, H, W]
    rendered_alphas = render_alphas[0].permute(2, 0, 1)

    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    screenspace_points = info["means2d"]

    ##### Convert into our own return format #####
    # concatenate RGB image with alpha image
    rendered_image = torch.cat((rendered_image, rendered_alphas), dim=0)
    depth_image = rendered_depth.squeeze(0)  # (1, H, W) -> (H, W)

    ##### Our normal rendering #####
    if render_normals:

        render_extras = {}

        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # (N, 3)

        # compute normal image (reference: GaussianShader)
        normal = pc.get_normal(dir_pp_normalized=dir_pp_normalized)
        normal_normed = normal * 0.5 + 0.5          # from [-1, 1] to [0, 1]
        render_extras["normal"] = normal_normed

        out_extras = {}
        for k in render_extras.keys():
            if render_extras[k] is None: continue
            render_colors = rasterization(
                means=means3D,    # [N, 3]
                quats=rotations,  # [N, 4]
                scales=scales,    # [N, 3]
                opacities=opacity.squeeze(-1),  # [N,]
                colors=render_extras[k],   # [N, 3] for normal
                viewmats=viewmat[None],  # [1, 4, 4]
                Ks=K[None],  # [1, 3, 3]
                backgrounds=None, # [1, 3]
                width=int(viewpoint_camera.image_width),
                height=int(viewpoint_camera.image_height),
                packed=False,
                sh_degree=None,
                render_mode='RGB+ED',
            )[0]
            image = render_colors[0].permute(2, 0, 1)[:3]   # [1, H, W, 4] -> [3, H, W]
            out_extras[k] = image

        for k in ["normal"]:
            if k in out_extras.keys():
                out_extras[k] = (out_extras[k] - 0.5) * 2. # from [0, 1] to [-1, 1]
    
        # normalize the normal map
        normal_image = out_extras["normal"]
        normal_image = normal_image.permute(1, 2, 0) # (H, W, 3)
        normal_image = torch.nn.functional.normalize(normal_image, p=2, dim=-1)
    else:
        normal_image = None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return_pkg = {
        "render": rendered_image,
        "depth": depth_image,
        "normal": normal_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }

    return return_pkg


def render_3dgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
```

gaussian_splatting/gaussian_renderer/network_gui.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import json
from ..scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

def init(wish_host, wish_port):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)

def try_connect():
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception as inst:
        pass
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))

def receive():
    message = read()

    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            do_shs_python = bool(message["shs_python"])
            do_rot_scale_python = bool(message["rot_scale_python"])
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
        except Exception as e:
            print("")
            traceback.print_exc()
            raise e
        return custom_cam, do_training, do_shs_python, do_rot_scale_python, keep_alive, scaling_modifier
    else:
        return None, None, None, None, None, None```

gaussian_splatting/generate_interp_poses.py
```python
import numpy as np
import scipy.interpolate
import pickle
import os


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def generate_interpolated_path(poses: np.ndarray,
                               n_interp: int,
                               spline_degree: int = 5,
                               smoothness: float = .03,
                               rot_weight: float = .1):
    """Creates a smooth spline path between input keyframe camera poses.
    Adapted from https://github.com/google-research/multinerf/blob/main/internal/camera_utils.py
    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points)


if __name__ == '__main__':
    root_dir = "./data/gaussian_data"
    for scene_name in sorted(os.listdir(root_dir)):
        scene_dir = os.path.join(root_dir, scene_name)
        print(f'Processing {scene_name}')
        camera_path = os.path.join(scene_dir, 'camera_meta.pkl')
        with open(camera_path, 'rb') as f:
            camera_meta = pickle.load(f)
        c2ws = camera_meta['c2ws']
        pose_0 = c2ws[0]
        pose_1 = c2ws[1]
        pose_2 = c2ws[2]
        n_interp = 50
        poses_01 = np.stack([pose_0, pose_1], 0)[:, :3, :]
        interp_poses_01 = generate_interpolated_path(poses_01, n_interp)
        poses_12 = np.stack([pose_1, pose_2], 0)[:, :3, :]
        interp_poses_12 = generate_interpolated_path(poses_12, n_interp)
        poses_20 = np.stack([pose_2, pose_0], 0)[:, :3, :]
        interp_poses_20 = generate_interpolated_path(poses_20, n_interp)
        interp_poses = np.concatenate([interp_poses_01, interp_poses_12, interp_poses_20], 0)
        output_poses = [np.vstack([pose, np.array([0, 0, 0, 1])]) for pose in interp_poses]
        pickle.dump(output_poses, open(os.path.join(scene_dir, 'interp_poses.pkl'), 'wb'))
        ```

gaussian_splatting/img2video.py
```python
import os
import imageio.v2 as imageio
# import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Convert images to video')
parser.add_argument('--image_folder', type=str, help='Path of image folder')
parser.add_argument('--video_path', type=str, help='Video filename')
parser.add_argument('--fps', type=int, default=15, help='Frame per second')
args = parser.parse_args()

image_folder = args.image_folder
video_path = args.video_path
fps = int(args.fps)

video_folder = os.path.dirname(video_path)
os.makedirs(video_folder, exist_ok=True)

images_path = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")])
if len(images_path) == 0:
    print("No images found in the folder")

frame_series = []
for image_path in images_path:
    image = imageio.imread(os.path.join(image_folder, image_path)).astype(np.uint8)
    h = image.shape[0] if image.shape[0] % 2 == 0 else image.shape[0] - 1
    w = image.shape[1] if image.shape[1] % 2 == 0 else image.shape[1] - 1
    frame_series.append(image[:h, :w])

imageio.mimsave(video_path, frame_series, fps=fps, macro_block_size=1)```

gaussian_splatting/lpipsPyTorch/__init__.py
```python
import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
```

gaussian_splatting/lpipsPyTorch/modules/lpips.py
```python
import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)
```

gaussian_splatting/lpipsPyTorch/modules/networks.py
```python
from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)
```

gaussian_splatting/lpipsPyTorch/modules/utils.py
```python
from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict
```

gaussian_splatting/rotation_utils.py
```python
import os
import torch
import torch.nn.functional as F
import einops
# from e3nn import o3


"""
Some functions are borrowed from PhysDreamer: https://github.com/a1600012888/PhysDreamer/blob/main/physdreamer/gaussian_3d/utils/rigid_body_utils.py
"""


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    from pytorch3d. Based on trace_method like: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L205
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def quternion_to_matrix(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    ret = torch.stack((ow, ox, oy, oz), -1)
    ret = standardize_quaternion(ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    from Pytorch3d
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# def transform_shs(shs_feat, rot_rotation_matrix):
#     """
#     Transform spherical harmonics features with rotation matrix
#     Borrowed from: https://github.com/graphdeco-inria/gaussian-splatting/issues/176#issuecomment-2060513169
#     TODO: this function has not been tested
#     """
#     #degree 1 transformation for now 
#     # frist_degree_shs = shs_feat[:, 0:1]
#     # permuting the last rgb to brg
#     mat = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).to(device=shs_feat.device).float()

#     rot_angles = o3._rotation.matrix_to_angles(rot_rotation_matrix.cpu())
#     #Construction coefficient
#     D_0 = o3.wigner_D(0, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
#     D_1 = o3.wigner_D(1, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
#     D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()
#     D_3 = o3.wigner_D(3, rot_angles[0], rot_angles[1], rot_angles[2]).cuda()

#     #rotation of the shs features
#     two_degree_shs = shs_feat[:, 0:3]
#     two_degree_shs = torch.matmul(two_degree_shs, mat)
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     two_degree_shs = torch.matmul(two_degree_shs, D_1)
#     # print(D_1.shape)
#     # print(two_degree_shs.shape)
#     # two_degree_shs = torch.einsum(
#     #         D_1,
#     #         two_degree_shs,
#     #         "... i j, ... j -> ... i",
#     #     )
#     two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     two_degree_shs = torch.matmul(two_degree_shs, torch.inverse(mat))
#     shs_feat[:, 0:3] = two_degree_shs

#     three_degree_shs = shs_feat[:, 3:8]
#     three_degree_shs = torch.matmul(three_degree_shs, mat)
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     three_degree_shs = torch.matmul(three_degree_shs, D_2)
#     # three_degree_shs = torch.einsum(
#     #         D_2,
#     #         three_degree_shs,
#     #         "... i j, ... j -> ... i",
#     #     )
#     three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     three_degree_shs = torch.matmul(three_degree_shs, torch.inverse(mat))
#     shs_feat[:, 3:8] = three_degree_shs

#     four_degree_shs = shs_feat[:, 8:15]
#     four_degree_shs = torch.matmul(four_degree_shs, mat)
#     four_degree_shs = einops.rearrange(four_degree_shs, 'n shs_num rgb -> n rgb shs_num')
#     four_degree_shs = torch.matmul(four_degree_shs, D_3)
#     # four_degree_shs = torch.einsum(
#     #         D_3,
#     #         four_degree_shs,
#     #         "... i j, ... j -> ... i",
#     #     )
#     four_degree_shs = einops.rearrange(four_degree_shs, 'n rgb shs_num -> n shs_num rgb')
#     four_degree_shs = torch.matmul(four_degree_shs, torch.inverse(mat))
#     shs_feat[:, 8:15] = four_degree_shs

#     return shs_feat```

gaussian_splatting/scene/__init__.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from ..utils.system_utils import searchForMaxIteration
from ..scene.dataset_readers import sceneLoadTypeCallbacks
from ..scene.gaussian_model import GaussianModel
from ..arguments import ModelParams
from ..utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import open3d as o3d
import numpy as np
import torch

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        elif os.path.exists(os.path.join(args.source_path, 'camera_meta.pkl')):
            print("Found metadata.json file, assuming customized QQTT dataset!")
            scene_info = sceneLoadTypeCallbacks["QQTT"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp, args.use_masks, args.gs_init_opt, args.pts_per_triangles, args.use_high_res)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        self.gaussians.isotropic = args.isotropic

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

        # Sample points from mesh or point cloud observation (used for regularization and gaussians removal)
        N_SAMPLES = 100_000
        mesh_path = os.path.join(args.source_path, 'shape_prior.glb')
        if os.path.exists(mesh_path):
            print(f"Sampling {N_SAMPLES} points from mesh")
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            sampled_points = np.asarray(mesh.sample_points_uniformly(number_of_points=N_SAMPLES).points)
        else:
            print(f"Sampled {N_SAMPLES} points from point cloud observation")
            pcd_path = os.path.join(args.source_path, 'observation.ply')
            pcd = o3d.io.read_point_cloud(pcd_path)
            xyz = np.asarray(pcd.points)
            num_points = min(xyz.shape[0], N_SAMPLES)
            sampled_points = xyz[np.random.choice(xyz.shape[0], num_points, replace=False)]
        self.mesh_sampled_points = torch.tensor(sampled_points, dtype=torch.float32, device="cuda")


    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
```

gaussian_splatting/scene/cameras.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from ..utils.graphics_utils import getWorld2View2, getProjectionMatrix
from ..utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False,
                 K=None, normal=None, depth=None, occ_mask=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.K = K

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            resized_image_rgb = PILtoTorch(image, resolution)
            gt_image = resized_image_rgb[:3, ...]
            self.alpha_mask = None
            if resized_image_rgb.shape[0] == 4:
                self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
            else: 
                self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

            if train_test_exp and is_test_view:
                if is_test_dataset:
                    self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
                else:
                    self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

            self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self.original_image.shape[2]
            self.image_height = self.original_image.shape[1]
        else:
            self.alpha_mask = None
            self.original_image = torch.zeros((3, resolution[1], resolution[0]), device=self.data_device)
            self.image_width = resolution[0]
            self.image_height = resolution[1]

        # extend additional alpha channel to original_image
        # self.original_image = torch.cat([self.original_image, torch.ones((1, self.image_height, self.image_width), device=self.data_device)], dim=0)

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.depth = depth.to(self.data_device) if depth is not None else None
        self.normal = normal.to(self.data_device) if normal is not None else None

        self.occ_mask = occ_mask.to(self.data_device) if occ_mask is not None else None
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

```

gaussian_splatting/scene/colmap_loader.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
```

gaussian_splatting/scene/dataset_readers.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from ..scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
)
from ..utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import SH2RGB
from ..scene.gaussian_model import BasicPointCloud

import pickle
import trimesh
import open3d as o3d
import cv2


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):

        # Extract all meshes from the scene
        meshes = []
        for name, geometry in scene_or_mesh.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)

        # Combine all meshes if there are multiple
        if len(meshes) > 1:
            combined_mesh = trimesh.util.concatenate(meshes)
        elif len(meshes) == 1:
            combined_mesh = meshes[0]
        else:
            raise ValueError("No valid meshes found in the GLB file")

        # Get model metadata
        metadata = {
            "vertices": combined_mesh.vertices.shape[0],
            "faces": combined_mesh.faces.shape[0],
            "bounds": combined_mesh.bounds.tolist(),
            "center_mass": combined_mesh.center_mass.tolist(),
            "is_watertight": combined_mesh.is_watertight,
            "original_scene": combined_mesh,  # Keep reference to original scene
        }

        mesh = combined_mesh
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool
    image: np.array = None
    normal: np.array = None
    depth: np.array = None
    K: np.array = None
    occ_mask: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depths_params,
    images_folder,
    depths_folder,
    test_cam_names_list,
):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split(".")[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = (
            os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png")
            if depths_folder != ""
            else ""
        )

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_path,
            image_name=image_name,
            depth_path=depth_path,
            width=width,
            height=height,
            is_test=image_name in test_cam_names_list,
        )
        cam_infos.append(cam_info)

    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb, normals=None):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, depths, eval, train_test_exp, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array(
                [depths_params[key]["scale"] for key in depths_params]
            )
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(
                f"Error: depth_params.json file not found at path '{depth_params_file}'."
            )
            sys.exit(1)
        except Exception as e:
            print(
                f"An unexpected error occurred when trying to open depth_params.json file: {e}"
            )
            sys.exit(1)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [
                name for idx, name in enumerate(cam_names) if idx % llffhold == 0
            ]
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), "r") as file:
                test_cam_names_list = [line.strip() for line in file]
    else:
        test_cam_names_list = []

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "",
        test_cam_names_list=test_cam_names_list,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


def readCamerasFromTransforms(
    path, transformsfile, depths_folder, white_background, is_test, extension=".png"
):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                1 - norm_data[:, :, 3:4]
            )
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            depth_path = (
                os.path.join(depths_folder, f"{image_name}.png")
                if depths_folder != ""
                else ""
            )

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=FovY,
                    FovX=FovX,
                    image_path=image_path,
                    image_name=image_name,
                    width=image.size[0],
                    height=image.size[1],
                    depth_path=depth_path,
                    depth_params=None,
                    is_test=is_test,
                )
            )

    return cam_infos


def readNerfSyntheticInfo(path, white_background, depths, eval, extension=".png"):

    depths_folder = os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", depths_folder, white_background, False, extension
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", depths_folder, white_background, True, extension
    )

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(
            points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
        )

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=True,
    )
    return scene_info


# def readQQTTSceneInfo(path, images, depths, eval, train_test_exp, use_masks=False, mesh_path=None):
#     # currently ignore parameter such as: images, depths, eval, train_test_exp

#     # read metadata
#     with open(os.path.join(path, 'metadata.json'), 'r') as f:
#         data = json.load(f)

#     # read cameras
#     intrinsics = np.array(data["intrinsics"])
#     WH = data["WH"]
#     width, height = WH
#     c2ws = pickle.load(open(os.path.join(path, 'calibrate.pkl'), 'rb'))
#     num_cam = len(intrinsics)
#     assert num_cam == len(c2ws), "Number of cameras and camera poses mismatched"

#     cam_infos_unsorted = []
#     for cam_i in range(num_cam):
#         c2w = c2ws[cam_i]
#         K = intrinsics[cam_i]

#         # get the world-to-camera transform and set R, T
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]

#         image_path = os.path.join(path, 'color', str(cam_i), '0.png')
#         image_name = f'cam{cam_i}_0'
#         image = Image.open(image_path) if os.path.exists(image_path) else None

#         # (Optional) use additional masks
#         if use_masks and image is not None:
#             mask_info_path = os.path.join(path, 'mask', f'mask_info_{cam_i}.json')
#             with open(mask_info_path, 'r') as f:
#                 mask_info = json.load(f)                                      # example: {"0": "hand", "1": "twine", "2": "hand"}
#             twine_id = [k for k, v in mask_info.items() if v == "twine"][0]   # assume only one twine
#             mask_path = os.path.join(path, 'mask', str(cam_i), twine_id, '0.png')
#             mask = np.array(Image.open(mask_path))
#             image_rgba = np.concatenate([np.array(image), mask[:, :, None]], axis=-1)
#             image = Image.fromarray(image_rgba)

#         # assume centered principal point at this moment, use K instead
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)

#         # load depth
#         depth_path = os.path.join(path, 'depth', str(cam_i), '0.npy')
#         depth = np.load(depth_path) / 1000.0 if os.path.exists(depth_path) else None  # in mm, convert to m

#         # load normal
#         # normal_path = os.path.join(path, 'normal_omnidata', str(cam_i), '0_normal.png')
#         normal_path = os.path.join(path, 'normal_metric3d', str(cam_i), '0.png')
#         normal = np.array(Image.open(normal_path)) if os.path.exists(normal_path) else None

#         if normal is not None:
#             normal = normal.astype(np.float32) / 255.0  # normalize to [0, 1]
#             normal = (normal - 0.5) * 2                 # normalize to [-1, 1]
#             W2C = getWorld2View2(R, T)
#             C2W = np.linalg.inv(W2C)
#             normal = normal @ C2W[:3, :3].T             # transform normal to world space

#         cam_infos_unsorted.append(CameraInfo(uid=cam_i, R=R, T=T, FovY=FovY, FovX=FovX,
#                             image_path=image_path, image_name=image_name,
#                             width=width, height=height, depth_path="", depth_params=None, is_test=False,
#                             K=K, image=image, normal=normal, depth=depth))

#     test_cam_infos = []
#     test_c2ws = pickle.load(open(os.path.join(path, 'interp_poses.pkl'), 'rb'))
#     for cam_i, c2w in enumerate(test_c2ws):
#         dummy_cam_id = 1
#         K = intrinsics[dummy_cam_id]
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]
#         image_path = os.path.join(path, 'color', str(dummy_cam_id), '0.png')
#         image_name = f'test_cam{cam_i}_0'
#         focal_length_x = K[0, 0]
#         focal_length_y = K[1, 1]
#         FovY = focal2fov(focal_length_y, height)
#         FovX = focal2fov(focal_length_x, width)
#         test_cam_infos.append(CameraInfo(uid=cam_i, R=R, T=T, FovY=FovY, FovX=FovX,
#                             image_path=image_path, image_name=image_name,
#                             width=width, height=height, depth_path="", depth_params=None, is_test=True,
#                             K=K, image=None, normal=None, depth=None))

#     cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
#     train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
#     test_cam_infos = [c for c in test_cam_infos if c.is_test]

#     nerf_normalization = getNerfppNorm(train_cam_infos)

#     # read point cloud
#     frame_idx = 0

#     # pcd_xyz_path = os.path.join(path, 'pcd', str(frame_idx), 'points.npy')
#     # pcd_color_path = os.path.join(path, 'pcd', str(frame_idx), 'colors.npy')  # [0-1]
#     # xyz = np.load(pcd_xyz_path)   # [N, 3]
#     # rgb = np.load(pcd_color_path) # [N, 3]

#     if use_masks:
#         data = np.load(os.path.join(path, 'pcd', str(frame_idx), 'first_frame_object.npz'))
#     else:
#         data = np.load(os.path.join(path, 'pcd', str(frame_idx), 'first_frame_total.npz'))
#     xyz = data['points']
#     rgb = data['colors']     # [0-1]
#     normals = np.zeros_like(xyz)

#     # sample init points from mesh if mesh_path is provided
#     if mesh_path:
#         print("Init points from mesh...", mesh_path)
#         xyz, rgb, normals = sample_pcd_from_mesh(mesh_path, POINT_PER_TRIANGLE=30)

#     pcd = BasicPointCloud(points=xyz, colors=rgb, normals=normals)

#     ply_path = os.path.join(path, 'pcd', str(frame_idx), 'points3D.ply')  # mimic other two dataloaders
#     storePly(ply_path, xyz, rgb)

#     # return scene info
#     scene_info = SceneInfo(point_cloud=pcd,
#                            train_cameras=train_cam_infos,
#                            test_cameras=test_cam_infos,
#                            nerf_normalization=nerf_normalization,
#                            ply_path=ply_path,
#                            is_nerf_synthetic=False)
#     return scene_info


def readQQTTSceneInfo(
    path,
    images,
    depths,
    eval,
    train_test_exp,
    use_masks=False,
    gs_init_opt="pcd",
    pts_per_triangles=30,
    use_high_res=False,
):
    # currently ignore parameter such as: images, depths, eval, train_test_exp

    # read metadata
    camera_info_path = os.path.join(path, "camera_meta.pkl")
    with open(camera_info_path, "rb") as f:
        camera_info = pickle.load(f)

    # read cameras
    intrinsics = [np.array(intr) for intr in camera_info["intrinsics"]]
    c2ws = camera_info["c2ws"]
    num_cam = len(intrinsics)
    assert num_cam == len(c2ws), "Number of cameras and camera poses mismatched"

    H, W = 480, 848  # fixed resolution

    if use_high_res:
        upsample = 4
        H = int(H * upsample)
        W = int(W * upsample)
        for intr in intrinsics:
            intr[0, 0] *= upsample
            intr[1, 1] *= upsample
            intr[0, 2] *= upsample
            intr[1, 2] *= upsample

    # get camera infos
    cam_infos_unsorted = []
    for cam_i in range(num_cam):
        c2w = c2ws[cam_i]
        K = intrinsics[cam_i]

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        image_path = (
            os.path.join(path, str(cam_i) + ".png")
            if not use_high_res
            else os.path.join(path, str(cam_i) + "_high.png")
        )
        image_name = f"cam{cam_i}"
        image = Image.open(image_path) if os.path.exists(image_path) else None

        # use additional masks
        if use_masks and image is not None:
            mask_path = (
                os.path.join(path, "mask_" + str(cam_i) + ".png")
                if not use_high_res
                else os.path.join(path, "mask_" + str(cam_i) + "_high.png")
            )
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]  # take the alpha channel
            image_rgba = np.concatenate([np.array(image), mask[:, :, None]], axis=-1)
            image = Image.fromarray(image_rgba)

        # this is dummy term for center principal point assumption (not used)
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, H)
        FovX = focal2fov(focal_length_x, W)

        # load depth
        depth_path = os.path.join(path, str(cam_i) + "_depth.npy")
        depth = (
            np.load(depth_path) / 1000.0 if os.path.exists(depth_path) else None
        )  # in mm, convert to m

        # load normal
        normal_path = os.path.join(path, str(cam_i) + "_normal_metric3d.png")
        normal = (
            np.array(Image.open(normal_path)) if os.path.exists(normal_path) else None
        )

        # load occ mask
        occ_mask_path = (
            os.path.join(path, "mask_human_" + str(cam_i) + ".png")
            if not use_high_res
            else os.path.join(path, "mask_human_" + str(cam_i) + "_high.png")
        )
        occ_mask = (
            np.array(Image.open(occ_mask_path))
            if os.path.exists(occ_mask_path)
            else None
        )
        if occ_mask is not None:
            if len(occ_mask.shape) == 3:
                occ_mask = occ_mask[:, :, -1]  # take the alpha channel
            occ_mask = occ_mask.astype(np.float32) / 255.0
            kernel_size = 8
            occ_mask = cv2.dilate(
                occ_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1
            )  # dilate to avoid boundary artifacts

        if normal is not None:
            normal = normal.astype(np.float32) / 255.0  # normalize to [0, 1]
            normal = (normal - 0.5) * 2  # normalize to [-1, 1]
            W2C = getWorld2View2(R, T)
            C2W = np.linalg.inv(W2C)
            normal = normal @ C2W[:3, :3].T  # transform normal to world space

        cam_infos_unsorted.append(
            CameraInfo(
                uid=cam_i,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=W,
                height=H,
                depth_path="",
                depth_params=None,
                is_test=False,
                K=K,
                image=image,
                normal=normal,
                depth=depth,
                occ_mask=occ_mask,
            )
        )

    test_cam_infos = []
    test_c2ws = pickle.load(open(os.path.join(path, "interp_poses.pkl"), "rb"))
    for cam_i, c2w in enumerate(test_c2ws):
        dummy_cam_id = 1
        K = intrinsics[dummy_cam_id]
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_path = os.path.join(path, "color", str(dummy_cam_id), "0.png")
        image_name = f"test_cam{cam_i}_0"
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, H)
        FovX = focal2fov(focal_length_x, W)
        test_cam_infos.append(
            CameraInfo(
                uid=cam_i,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image_path=image_path,
                image_name=image_name,
                width=W,
                height=H,
                depth_path="",
                depth_params=None,
                is_test=True,
                K=K,
                image=None,
                normal=None,
                depth=None,
            )
        )

    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in test_cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # read point cloud ('pcd', 'mesh', 'hybrid')
    all_xyz, all_rgb, all_normals = [], [], []
    if gs_init_opt in ["pcd", "hybrid"]:
        print("Init points from pcd...")
        pcd_path = os.path.join(path, "observation.ply")
        if os.path.exists(pcd_path):
            pcd = o3d.io.read_point_cloud(pcd_path)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_normals.append(np.zeros((xyz.shape[0], 3)))

    if gs_init_opt in ["mesh", "hybrid"]:
        print("Init points from mesh...")
        mesh_path = os.path.join(path, "shape_prior.glb")
        if os.path.exists(mesh_path):
            xyz, rgb, normals = sample_pcd_from_mesh(
                mesh_path, POINT_PER_TRIANGLE=pts_per_triangles
            )
            all_xyz.append(xyz)
            all_rgb.append(rgb)
            all_normals.append(normals)

    assert len(all_xyz) > 0, "No point cloud or mesh found for initialization"

    all_xyz = np.concatenate(all_xyz, axis=0)
    all_rgb = np.concatenate(all_rgb, axis=0)
    all_normals = np.concatenate(all_normals, axis=0)

    pcd = BasicPointCloud(points=all_xyz, colors=all_rgb, normals=all_normals)

    ply_path = os.path.join(path, "points3D.ply")  # mimic other two dataloaders
    storePly(ply_path, all_xyz, all_rgb, all_normals)

    # return scene info
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        is_nerf_synthetic=False,
    )
    return scene_info


def sample_pcd_from_mesh(mesh_path, POINT_PER_TRIANGLE=5):
    """
    Sample points from uv-textured mesh
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    has_uv_texture = np.asarray(mesh.triangle_uvs).shape[0] != 0
    if has_uv_texture:
        uvs = np.asarray(mesh.triangle_uvs).reshape(-1, 3, 2)
        texture = np.asarray(mesh.textures[0])
    else:
        vertex_colors = np.asarray(mesh.vertex_colors)
        if vertex_colors.shape[0] != vertices.shape[0]:
            raise ValueError("Mesh has no texture or valid vertex colors.")

    mesh.compute_triangle_normals()
    triangles_normals = np.asarray(mesh.triangle_normals)

    n_triangles = triangles.shape[0]
    total_sample_points = n_triangles * POINT_PER_TRIANGLE

    sampled_points = np.zeros((total_sample_points, 3), dtype=np.float32)
    sampled_colors = np.zeros((total_sample_points, 3), dtype=np.float32)
    sampled_normals = np.zeros((total_sample_points, 3), dtype=np.float32)

    for i in range(n_triangles):
        tri_vertices = vertices[triangles[i]]

        # generate barycentric coordinates
        r1 = np.random.rand(POINT_PER_TRIANGLE)
        r2 = np.random.rand(POINT_PER_TRIANGLE)
        u = 1 - np.sqrt(r1)
        v = r2 * np.sqrt(r1)
        w = 1 - u - v
        barycentric = np.vstack((u, v, w)).T
        points = np.dot(barycentric, tri_vertices)

        if has_uv_texture:
            tri_uvs = uvs[i]
            uv_points = np.dot(barycentric, tri_uvs)
            # convert uv to texture coordinates
            px = np.clip(
                (uv_points[:, 0] * texture.shape[1]).astype(int),
                0,
                texture.shape[1] - 1,
            )
            py = np.clip(
                (uv_points[:, 1] * texture.shape[0]).astype(int),
                0,
                texture.shape[0] - 1,
            )
            colors = texture[py, px] / 255.0
        else:
            # interpolate vertex colors using barycentric coords
            tri_colors = vertex_colors[triangles[i]]
            colors = np.dot(barycentric, tri_colors)

        normals = triangles_normals[i]

        start_idx = i * POINT_PER_TRIANGLE
        end_idx = (i + 1) * POINT_PER_TRIANGLE
        sampled_points[start_idx:end_idx] = points
        sampled_colors[start_idx:end_idx] = colors
        sampled_normals[start_idx:end_idx] = normals

    return sampled_points, sampled_colors, sampled_normals


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "QQTT": readQQTTSceneInfo,
}
```

gaussian_splatting/scene/gaussian_model.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from ..utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from ..utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from ..utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from ..utils.graphics_utils import BasicPointCloud
from ..utils.general_utils import strip_symmetric, build_scaling_rotation, get_minimum_axis, flip_align_view

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.isotropic = False

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.isotropic:
            return self.scaling_activation(self._scaling).repeat(1, 3)
        else:
            return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def get_normal(self, dir_pp_normalized=None):
        normal_axis = self.get_minimum_axis
        normal_axis, positive = flip_align_view(normal_axis, dir_pp_normalized)
        normal = normal_axis / normal_axis.norm(dim=1, keepdim=True) # (N, 3)
        return normal
    
    @property
    def get_minimum_axis(self):
        return get_minimum_axis(self.get_scaling, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        if self.isotropic:
            scales = torch.log(torch.sqrt(dist2))[...,None]
        else:
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                # print(f"Extending {group['name']} from {group['params'][0].shape} to {extension_tensor.shape}")

                # print("===== ", group['name'], " =====")
                # print(stored_state['exp_avg'].shape, extension_tensor.shape)

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        if self.isotropic:
            scaling_tensor = self.get_scaling[selected_pts_mask]
            # print(scaling_tensor.shape)
            if scaling_tensor.shape[0] == 0:
                new_scaling = scaling_tensor[:, :1]  # Ensures shape (0, 1) for N = 0
            else:
                new_scaling = self.scaling_inverse_activation(scaling_tensor[:, 0].unsqueeze(-1).repeat(N, 1) / (0.8 * N))
                # print("New scaling 2: ", new_scaling.shape)
        else:
            new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height, use_gsplat=True, use_absgrad=False):
        if use_gsplat:
            # https://github.com/liruilong940607/gaussian-splatting/commit/258123ab8f8d52038da862c936bd413dc0b32e4d
            # grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
            if use_absgrad:
                # grads = info["means2d"].absgrad.clone()
                grad = viewspace_point_tensor.absgrad.squeeze(0) # [N, 2]
            else:
                # grads = info["means2d"].grad.clone()
                grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
            # Normalize the gradient to [-1, 1] screen size
            grad[:, 0] *= width * 0.5
            grad[:, 1] *= height * 0.5
            self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1```

gaussian_splatting/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )

```

gaussian_splatting/submodules/diff-gaussian-rasterization/setup.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

gaussian_splatting/submodules/fused-ssim/fused_ssim/__init__.py
```python
from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()
```

gaussian_splatting/submodules/fused-ssim/setup.py
```python
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
            "ssim.cu",
            "ext.cpp"])
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

gaussian_splatting/submodules/fused-ssim/tests/genplot.py
```python
import torch
from fused_ssim import fused_ssim
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt
import numpy as np
import time
import os

plt.style.use('ggplot')
gpu = torch.cuda.get_device_name()

if __name__ == "__main__":
    torch.manual_seed(0)

    B, CH = 5, 1
    dimensions = list(range(50, 1550, 50))
    iterations = 50

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    pm_ssim = SSIM(data_range=1.0, channel=CH)

    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d], device="cuda")
            img2_og = torch.rand([B, CH, d, d], device="cuda")

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

        begin = time.time()
        for _ in range(iterations):
            pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            pm_ssim_val.backward()
        torch.cuda.synchronize()
        end = time.time()
        data["pytorch_mssim"].append((end - begin) / iterations * 1000)

        begin = time.time()
        for _ in range(iterations):
            mine_ssim_val_same = fused_ssim(img1_mine_same, img2_mine_same)
            mine_ssim_val_same.backward()
        torch.cuda.synchronize()
        end = time.time()
        data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 2) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one training iteration (ms).")
    plt.title(f"Training Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", "training_time.png"), dpi=300)

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    plt.clf()
    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d], device="cuda")
            img2_og = torch.rand([B, CH, d, d], device="cuda")

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

            begin = time.time()
            for _ in range(iterations):
                pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            torch.cuda.synchronize()
            end = time.time()
            data["pytorch_mssim"].append((end - begin) / iterations * 1000)

            begin = time.time()
            for _ in range(iterations):
                mine_ssim_val_same = fused_ssim(img1_mine_same, img2_mine_same, train=False)
            torch.cuda.synchronize()
            end = time.time()
            data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 2) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one inference iteration (ms).")
    plt.title(f"Inference Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", "inference_time.png"), dpi=300)
```

gaussian_splatting/submodules/fused-ssim/tests/train_image.py
```python
import torch
import numpy as np
import os
from PIL import Image
from fused_ssim import fused_ssim

gt_image = torch.tensor(np.array(Image.open(os.path.join("..", "images", "albert.jpg"))), dtype=torch.float32, device="cuda").unsqueeze(0).unsqueeze(0) / 255.0
pred_image = torch.nn.Parameter(torch.rand_like(gt_image))

with torch.no_grad():
    ssim_value = fused_ssim(pred_image, gt_image, train=False)
    print("Starting with SSIM value:", ssim_value)


optimizer = torch.optim.Adam([pred_image])

while ssim_value < 0.9999:
    optimizer.zero_grad()
    loss = 1.0 - fused_ssim(pred_image, gt_image)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        ssim_value = fused_ssim(pred_image, gt_image, train=False)
        print("SSIM value:", ssim_value)

pred_image = (pred_image * 255.0).squeeze(0).squeeze(0)
to_save = pred_image.detach().cpu().numpy().astype(np.uint8)
Image.fromarray(to_save).save(os.path.join("..", "images", "predicted.jpg"))
```

gaussian_splatting/submodules/simple-knn/setup.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

gaussian_splatting/utils/camera_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from ..scene.cameras import Camera
import numpy as np
from ..utils.graphics_utils import fov2focal
from ..utils.general_utils import PILtoTorch
from PIL import Image
import cv2
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    # image = Image.open(cam_info.image_path)
    image = cam_info.image

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    # orig_w, orig_h = image.size
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        K = cam_info.K / (resolution_scale * args.resolution)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
        K = cam_info.K / scale

    # resize depth and normal
    depth = None
    normal = None
    if cam_info.depth is not None:
        depth = torch.from_numpy(cam_info.depth).unsqueeze(0).unsqueeze(0)
        depth = torch.nn.functional.interpolate(depth, (resolution[1], resolution[0]), mode='bilinear', align_corners=True)
        depth = depth.squeeze(0).squeeze(0)          # (H, W)
    if cam_info.normal is not None:
        normal = torch.from_numpy(cam_info.normal).permute(2, 0, 1).unsqueeze(0)
        normal = torch.nn.functional.interpolate(normal, (resolution[1], resolution[0]), mode='nearest')
        normal = normal.squeeze(0).permute(1, 2, 0)  # (H, W, 3)

    occ_mask = None
    if cam_info.occ_mask is not None:
        occ_mask = torch.from_numpy(cam_info.occ_mask).unsqueeze(0).unsqueeze(0)
        occ_mask = torch.nn.functional.interpolate(occ_mask, (resolution[1], resolution[0]), mode='nearest')
        occ_mask = occ_mask.squeeze(0).squeeze(0)    # (H, W)
    
    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  K=cam_info.K, normal=normal, depth=depth, occ_mask=occ_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry```

gaussian_splatting/utils/general_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import torch
import sys
from datetime import datetime
import numpy as np
import random
from typing import Callable, Tuple
from PIL import Image

def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))

def PILtoTorch(pil_image: Image.Image, resolution: Tuple[int, int]) -> torch.Tensor:
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1000000,
) -> Callable[[int], float]:
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L: torch.Tensor) -> torch.Tensor:
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym: torch.Tensor) -> torch.Tensor:
    return strip_lowerdiag(sym)

def build_rotation(r: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent: bool) -> None:
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def get_minimum_axis(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,:,0]
    return x_axis


def flip_align_view(normal: torch.Tensor, viewdir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True)          # (N, 1)
    non_flip = dotprod >= 0                                # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1) # (N, 3)
    return normal_flipped, non_flip
```

gaussian_splatting/utils/graphics_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray

def geom_transform_points(points: torch.Tensor, transf_matrix: torch.Tensor) -> torch.Tensor:
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(
    R: np.ndarray,
    t: np.ndarray,
    translate: np.ndarray = np.array([0.0, 0.0, 0.0]),
    scale: float = 1.0,
) -> np.ndarray:
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear: float, zfar: float, fovX: float, fovY: float) -> torch.Tensor:
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal: float, pixels: float) -> float:
    return 2 * math.atan(pixels / (2 * focal))
```

gaussian_splatting/utils/image_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import Tensor


def mse(img1: Tensor, img2: Tensor) -> Tensor:
    """Return mean squared error between two images."""
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1: Tensor, img2: Tensor) -> Tensor:
    """Return peak signal-to-noise ratio between two images."""
    mse = (((img1 - img2)) ** 2).reshape(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
```

gaussian_splatting/utils/loss_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()


def normal_loss(network_output, gt, alpha_mask=None):
    '''
    Use normal to regularize the normal prediction
    Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
    '''
    assert network_output.shape[-1] == 3                                 # expected shape: (H, W, 3)
    normal_pred = F.normalize(network_output, p=2, dim=-1)               # (H, W, 3)
    normal_gt = F.normalize(gt, p=2, dim=-1)                             # (H, W, 3)
    
    if alpha_mask is not None:
        # mask = alpha_mask.squeeze().unsqueeze(-1)
        mask = (alpha_mask.squeeze() > 0.5).float().unsqueeze(-1)
        # normal_pred = normal_pred * mask
        normal_gt = normal_gt * mask
    
    l1_loss = torch.abs(normal_pred - normal_gt).mean()                  # L1 loss (H, W, 3)
    cos_loss = -torch.sum(normal_pred * normal_gt, axis=-1).mean()       # Cosine similarity loss (H, W, 3)
    return l1_loss + 0.1 * cos_loss


def depth_loss(network_output, gt, alpha_mask=None):
    '''
    Use disparity to regularize the depth prediction
    '''
    # valid_mask = (gt > 0).float()
    assert (gt < 0.0).sum() == 0, "Depth map should be non-negative"

    if alpha_mask is not None:
        # mask = alpha_mask.squeeze()
        mask = (alpha_mask.squeeze() > 0.5).float()
        # network_output = network_output * mask
        gt = gt * mask
        
    # network_output = network_output * valid_mask
    # gt = gt * valid_mask

    # disp_pred = 1.0 / (network_output + 1e-6)
    # disp_gt = 1.0 / (gt + 1e-6)
    # l1_loss = torch.abs(disp_pred - disp_gt).mean()

    l1_loss = torch.abs(network_output - gt).mean()
    
    return l1_loss


def anisotropic_loss(gaussians_scale, r=3):
    '''
    Use to regularize gaussians size to be isotropic (avoid over-stretching gaussians)
    Reference from PhysGaussian (https://arxiv.org/pdf/2311.12198)
    '''
    # L_aniso = mean( max( max(scale)/min(scale), r ) - r)
    eps = 1e-6
    max_scale = torch.max(gaussians_scale, dim=-1).values
    min_scale = torch.min(gaussians_scale, dim=-1).values
    return torch.mean(torch.clamp(max_scale / (min_scale + eps), min=r) - r)```

gaussian_splatting/utils/make_depth_scale.py
```python
import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images_metas[key].point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument('--depths_dir', default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument('--model_type', default="bin")
    args = parser.parse_args()


    cam_intrinsics, images_metas, points3d = read_model(os.path.join(args.base_dir, "sparse", "0"), ext=f".{args.model_type}")

    pts_indices = np.array([points3d[key].id for key in points3d])
    pts_xyzs = np.array([points3d[key].xyz for key in points3d])
    points3d_ordered = np.zeros([pts_indices.max()+1, 3])
    points3d_ordered[pts_indices] = pts_xyzs

    # depth_param_list = [get_scales(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas]
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, images_metas, points3d_ordered, args) for key in images_metas
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    with open(f"{args.base_dir}/sparse/0/depth_params.json", "w") as f:
        json.dump(depth_params, f, indent=2)

    print(0)
```

gaussian_splatting/utils/read_write_model.py
```python
# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import collections
import numpy as np
import struct
import argparse


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# def main():
#     parser = argparse.ArgumentParser(
#         description="Read and write COLMAP binary and text models"
#     )
#     parser.add_argument("--input_model", help="path to input model folder")
#     parser.add_argument(
#         "--input_format",
#         choices=[".bin", ".txt"],
#         help="input model format",
#         default="",
#     )
#     parser.add_argument("--output_model", help="path to output model folder")
#     parser.add_argument(
#         "--output_format",
#         choices=[".bin", ".txt"],
#         help="outut model format",
#         default=".txt",
#     )
#     args = parser.parse_args()

#     cameras, images, points3D = read_model(
#         path=args.input_model, ext=args.input_format
#     )

#     print("num_cameras:", len(cameras))
#     print("num_images:", len(images))
#     print("num_points3D:", len(points3D))

#     if args.output_model is not None:
#         write_model(
#             cameras,
#             images,
#             points3D,
#             path=args.output_model,
#             ext=args.output_format,
#         )


# if __name__ == "__main__":
#     main()
```

gaussian_splatting/utils/sh_utils.py
```python
#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5```

gaussian_splatting/utils/system_utils.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

from errno import EEXIST
from os import makedirs, path
import os
from typing import Any

def mkdir_p(folder_path: str) -> None:
    """Create a directory similar to ``mkdir -p``."""
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder: str) -> int:
    """Return the maximum iteration index found in ``folder``."""
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
```

gs_render.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
from kornia import create_meshgrid
import copy
import pytorch3d
import pytorch3d.ops as ops


def render_set(
    model_path: str,
    name: str,
    iteration: int,
    views: list,
    gaussians: GaussianModel,
    pipeline: PipelineParams,
    background: torch.Tensor,
    train_test_exp: bool,
    separate_sh: bool,
    disable_sh: bool = False,
) -> None:
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # TODO: temporary debug for demo
    # scene_name = model_path.split('/')[-2]
    # render_path = os.path.join('./output_tmp_for_sydney', scene_name, "renders")
    # gts_path = os.path.join('./output_tmp_for_sydney', scene_name, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        if disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            results = render(view, gaussians, pipeline, background, override_color=override_color, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        else:
            results = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
        rendering = results["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
) -> None:
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # TODO: quick demo purpose (remove later)
        # # sub-sample the gaussians
        # n_subsample = 1000
        # idx = torch.randperm(gaussians._xyz.size(0))[:n_subsample]
        # gaussians._xyz = gaussians._xyz[idx]
        # gaussians._features_dc = gaussians._features_dc[idx]
        # gaussians._features_rest = gaussians._features_rest[idx]
        # gaussians._scaling = gaussians._scaling[idx]
        # gaussians._rotation = gaussians._rotation[idx]
        # gaussians._opacity = gaussians._opacity[idx]
        # # set the scale of the gaussians
        # scale = 0.01
        # gaussians._scaling = gaussians.scaling_inverse_activation(torch.ones_like(gaussians._scaling) * scale)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, disable_sh=dataset.disable_sh)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh, disable_sh=dataset.disable_sh)


def get_ray_directions(
    H: int,
    W: int,
    K: torch.Tensor,
    device: str = "cuda",
    random: bool = False,
    return_uv: bool = False,
    flatten: bool = True,
    anti_aliasing_factor: float = 1.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    if anti_aliasing_factor > 1.0:
        H = int(H * anti_aliasing_factor) 
        W = int(W * anti_aliasing_factor) 
        K *= anti_aliasing_factor
        K[2, 2] = 1
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    if return_uv:
        return directions, grid
    return directions


def remove_gaussians_with_mask(
    gaussians: GaussianModel, views: list
) -> GaussianModel:
    gaussians_xyz = gaussians._xyz.detach()
    gaussians_view_counter = torch.zeros(gaussians_xyz.shape[0], dtype=torch.int32, device='cuda')
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            H, W = view.image_height, view.image_width
            K = view.K
            R, T = view.R, view.T

            # Create the World-to-Camera transformation matrix
            W2C = np.zeros((4, 4))
            W2C[:3, :3] = R.transpose()
            W2C[:3, 3] = T
            W2C[3, 3] = 1.0
            W2C = torch.tensor(W2C, dtype=torch.float32, device='cuda')

            # Transform gaussians' xyz coordinates to the camera space
            xyz = torch.cat([gaussians_xyz, torch.ones(gaussians_xyz.size(0), 1, device='cuda')], dim=1)
            xyz = torch.matmul(xyz, W2C.T)
            xyz = xyz[:, :3]
            xyz = xyz / xyz[:, 2].unsqueeze(1)  # Normalize by z-coordinate

            # Project to image plane
            uv = torch.matmul(xyz, torch.FloatTensor(K).to("cuda").T)
            uv = uv[:, :2].round().long()   # Convert to integer pixel coordinates

            # Check if (u, v) coordinates are within the image bounds
            alpha_mask = view.alpha_mask.squeeze(0)    # Assuming mask is a 2D tensor on CUDA with shape [H, W]
            valid_uv = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

            # Filter valid coordinates and check mask values
            for i, (u, v) in enumerate(uv):
                if valid_uv[i] and alpha_mask[v, u] > 0:  # Mask value > 0 implies it lies within the mask region
                    gaussians_view_counter[i] += 1
        
        # Remove the gaussians that are visible in a frequency of less than 50% of the views
        VIEW_THRESHOLD = 1.0
        mask3d = gaussians_view_counter >= len(views) * VIEW_THRESHOLD
        print(f"Removing {len(mask3d) - mask3d.sum()} gaussians not visible in {VIEW_THRESHOLD * 100}% of the views")
        new_gaussians = copy.deepcopy(gaussians)
        new_gaussians._xyz = gaussians._xyz[mask3d]
        new_gaussians._features_dc = gaussians._features_dc[mask3d]
        new_gaussians._features_rest = gaussians._features_rest[mask3d]
        new_gaussians._scaling = gaussians._scaling[mask3d]
        new_gaussians._rotation = gaussians._rotation[mask3d]
        new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def remove_gaussians_with_low_opacity(
    gaussians: GaussianModel, opacity_threshold: float = 0.1
) -> GaussianModel:

    opacity = gaussians.get_opacity.squeeze(-1)
    mask3d = opacity > opacity_threshold
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with opacity < 0.1")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


def remove_gaussians_with_point_mesh_distance(
    gaussians: GaussianModel,
    mesh_sampled_points: torch.Tensor,
    dist_threshold: float = 0.1,
) -> GaussianModel:
    '''
    Remove gaussians that are far from the mesh

    Args:
        gaussians (GaussianModel): Gaussian model
        mesh_sampled_points (Tensor): Sampled points from the mesh
        dist_threshold (float): Distance threshold (in meters) to remove the gaussians
    '''

    gaussians_xyz = gaussians._xyz.detach()
    # dists_knn = ops.knn_points(gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, norm=2)
    dists_bq = ops.ball_query(gaussians_xyz.unsqueeze(0), mesh_sampled_points.unsqueeze(0), K=1, radius=dist_threshold)
    mask3d = (dists_bq[1].squeeze(0) != -1).squeeze(-1)
    print(f"Removing {len(mask3d) - mask3d.sum()} gaussians with distance < {dist_threshold}")

    new_gaussians = copy.deepcopy(gaussians)
    new_gaussians._xyz = gaussians._xyz[mask3d]
    new_gaussians._features_dc = gaussians._features_dc[mask3d]
    new_gaussians._features_rest = gaussians._features_rest[mask3d]
    new_gaussians._scaling = gaussians._scaling[mask3d]
    new_gaussians._rotation = gaussians._rotation[mask3d]
    new_gaussians._opacity = gaussians._opacity[mask3d]

    return new_gaussians


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.remove_gaussians)```

gs_render_dynamics.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from gaussian_splatting.scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_splatting.gaussian_renderer import render
import torchvision
from gaussian_splatting.utils.general_utils import safe_state
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_splatting.gaussian_renderer import GaussianModel

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

import numpy as np
from kornia import create_meshgrid
import copy
from gs_render import (
    remove_gaussians_with_mask,
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.dynamic_utils import (
    interpolate_motions,
    create_relation_matrix,
    knn_weights,
    get_topk_indices,
    quat2mat,
    mat2quat,
)
import pickle


def render_set(
    output_path: str,
    name: str,
    views: list,
    gaussians_list: list,
    pipeline: PipelineParams,
    background: torch.Tensor,
    train_test_exp: bool,
    separate_sh: bool,
    disable_sh: bool = False,
) -> None:

    render_path = os.path.join(output_path, name)
    makedirs(render_path, exist_ok=True)

    # view_indices = [0, 25, 50, 75, 100, 125]
    view_indices = [0, 50, 100]
    selected_views = [views[i] for i in view_indices]

    for idx, view in enumerate(tqdm(selected_views, desc="Rendering progress")):

        # view_idx = view_indices[idx]
        # view_render_path = os.path.join(render_path, '{0:05d}'.format(view_idx))
        view_render_path = os.path.join(render_path, f"{idx}")
        makedirs(view_render_path, exist_ok=True)

        for frame_idx, gaussians in enumerate(gaussians_list):

            if disable_sh:
                override_color = gaussians.get_features_dc.squeeze()
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    override_color=override_color,
                    use_trained_exp=train_test_exp,
                    separate_sh=separate_sh,
                )
            else:
                results = render(
                    view,
                    gaussians,
                    pipeline,
                    background,
                    use_trained_exp=train_test_exp,
                    separate_sh=separate_sh,
                )

            rendering = results["render"]

            torchvision.utils.save_image(
                rendering,
                os.path.join(view_render_path, "{0:05d}".format(frame_idx) + ".png"),
            )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    separate_sh: bool,
    remove_gaussians: bool = False,
    name: str = "dynamic",
    output_dir: str = "./gaussian_output_dynamic",
) -> None:
    with torch.no_grad():
        output_path = output_dir

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # remove gaussians that are outside the mask
        if remove_gaussians:
            gaussians = remove_gaussians_with_mask(gaussians, scene.getTrainCameras())

        # remove gaussians that are low opacity
        gaussians = remove_gaussians_with_low_opacity(gaussians)

        # remove gaussians that are far from the mesh
        # gaussians = remove_gaussians_with_point_mesh_distance(gaussians, scene.mesh_sampled_points, dist_threshold=0.01)

        # rollout
        exp_name = dataset.source_path.split("/")[-1]
        ctrl_pts_path = f"./experiments/{exp_name}/inference.pkl"
        with open(ctrl_pts_path, "rb") as f:
            ctrl_pts = pickle.load(f)  # (n_frames, n_ctrl_pts, 3) ndarray
        ctrl_pts = torch.tensor(ctrl_pts, dtype=torch.float32, device="cuda")

        xyz_0 = gaussians.get_xyz
        rgb_0 = gaussians.get_features_dc.squeeze(1)
        quat_0 = gaussians.get_rotation
        opa_0 = gaussians.get_opacity
        scale_0 = gaussians.get_scaling

        # print(gaussians.get_features_dc.shape)   # (N, 1, 3)
        # print(gaussians.get_features_rest.shape) # (N, 15, 3)

        print("===== Number of steps: ", ctrl_pts.shape[0])
        print("===== Number of control points: ", ctrl_pts.shape[1])
        print("===== Number of gaussians: ", gaussians.get_xyz.shape[0])

        n_steps = ctrl_pts.shape[0]

        # rollout
        xyz, rgb, quat, opa = rollout(xyz_0, rgb_0, quat_0, opa_0, ctrl_pts, n_steps)

        # interpolate smoothly
        change_points = (
            (xyz - torch.cat([xyz[0:1], xyz[:-1]], dim=0))
            .norm(dim=-1)
            .sum(dim=-1)
            .nonzero()
            .squeeze(1)
        )
        change_points = torch.cat([torch.tensor([0]), change_points])
        for i in range(1, len(change_points)):
            start = change_points[i - 1]
            end = change_points[i]
            if end - start < 2:  # 0 or 1
                continue
            xyz[start:end] = torch.lerp(
                xyz[start][None],
                xyz[end][None],
                torch.linspace(0, 1, end - start + 1).to(xyz.device)[:, None, None],
            )[:-1]
            rgb[start:end] = torch.lerp(
                rgb[start][None],
                rgb[end][None],
                torch.linspace(0, 1, end - start + 1).to(rgb.device)[:, None, None],
            )[:-1]
            quat[start:end] = torch.lerp(
                quat[start][None],
                quat[end][None],
                torch.linspace(0, 1, end - start + 1).to(quat.device)[:, None, None],
            )[:-1]
            opa[start:end] = torch.lerp(
                opa[start][None],
                opa[end][None],
                torch.linspace(0, 1, end - start + 1).to(opa.device)[:, None, None],
            )[:-1]
        quat = torch.nn.functional.normalize(quat, dim=-1)

        gaussians_list = []
        for i in range(n_steps):
            gaussians_i = copy.deepcopy(gaussians)
            gaussians_i._xyz = xyz[i].to("cuda")
            gaussians_i._features_dc = rgb[i].unsqueeze(1).to("cuda")
            gaussians_i._rotation = quat[i].to("cuda")
            gaussians_i._opacity = gaussians_i.inverse_opacity_activation(opa[i]).to(
                "cuda"
            )
            gaussians_i._scaling = gaussians._scaling
            gaussians_list.append(gaussians_i)

        views = scene.getTestCameras()

        render_set(
            output_path,
            name,
            views,
            gaussians_list,
            pipeline,
            background,
            dataset.train_test_exp,
            separate_sh,
            disable_sh=dataset.disable_sh,
        )


def rollout(
    xyz_0: torch.Tensor,
    rgb_0: torch.Tensor,
    quat_0: torch.Tensor,
    opa_0: torch.Tensor,
    ctrl_pts: torch.Tensor,
    n_steps: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # store results
    xyz = xyz_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    rgb = rgb_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 3)
    quat = quat_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 4)
    opa = opa_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_gaussians, 1)

    # init relation matrix
    init_particle_pos = ctrl_pts[0]
    relations = get_topk_indices(init_particle_pos, K=16)

    all_pos = xyz_0
    all_rot = quat_0

    for i in tqdm(range(1, n_steps), desc="Rollout progress", dynamic_ncols=True):

        prev_particle_pos = ctrl_pts[i - 1]
        cur_particle_pos = ctrl_pts[i]

        # relations = get_topk_indices(prev_particle_pos, K=16)

        # interpolate all_pos and particle_pos
        chunk_size = 20_000
        num_chunks = (len(all_pos) + chunk_size - 1) // chunk_size
        for j in range(num_chunks):
            start = j * chunk_size
            end = min((j + 1) * chunk_size, len(all_pos))
            all_pos_chunk = all_pos[start:end]
            all_rot_chunk = all_rot[start:end]
            weights = knn_weights(prev_particle_pos, all_pos_chunk, K=16)
            all_pos_chunk, all_rot_chunk, _ = interpolate_motions(
                bones=prev_particle_pos,
                motions=cur_particle_pos - prev_particle_pos,
                relations=relations,
                weights=weights,
                xyz=all_pos_chunk,
                quat=all_rot_chunk,
            )
            all_pos[start:end] = all_pos_chunk
            all_rot[start:end] = all_rot_chunk

        quat[i] = all_rot.cpu()
        xyz[i] = all_pos.cpu()
        rgb[i] = rgb[i - 1].clone()
        opa[i] = opa[i - 1].clone()

    return xyz, rgb, quat, opa


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--remove_gaussians", action="store_true")
    parser.add_argument("--name", default="sceneA", type=str)
    parser.add_argument("--output_dir", default="./gaussian_output_dynamic", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        SPARSE_ADAM_AVAILABLE,
        args.remove_gaussians,
        args.name,
        args.output_dir,
    )

    with open("./rendering_finished_dynamic.txt", "a") as f:
        f.write("Rendering finished of " + args.name + "\n")
```

gs_train.py
```python
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, depth_loss, normal_loss, anisotropic_loss
from gaussian_splatting.gaussian_renderer import render, network_gui
import sys
from gaussian_splatting.scene import Scene, GaussianModel
from gaussian_splatting.utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from gaussian_splatting.utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from gaussian_splatting.arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations: list[int],
    saving_iterations: list[int],
    checkpoint_iterations: list[int],
    checkpoint: str | None,
    debug_from: int,
) -> None:

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if dataset.disable_sh:
            override_color = gaussians.get_features_dc.squeeze()
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=override_color, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        
        image, depth, normal, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], \
            render_pkg["depth"], \
            render_pkg["normal"], \
            render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], \
            render_pkg["radii"]
        
        pred_seg = image[3:, ...]
        image = image[:3, ...]
        gt_image = viewpoint_cam.original_image.cuda()

        # Mask out occluded regions
        if viewpoint_cam.occ_mask is not None:

            occ_mask = viewpoint_cam.occ_mask.cuda()
            inv_occ_mask = 1.0 - occ_mask
            
            # Expand inv_occ_mask to match each tensor shape
            image *= inv_occ_mask.unsqueeze(0)        # Shape: [3, 480, 848]
            # gt_image *= inv_occ_mask.unsqueeze(0)     # Shape: [3, 480, 848]
            pred_seg *= inv_occ_mask.unsqueeze(0)     # Shape: [1, 480, 848]
            depth *= inv_occ_mask                    # Shape: [480, 848]
            if normal is not None:
                normal *= inv_occ_mask.unsqueeze(-1)      # Shape: [480, 848, 3]

        # Loss
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            # image *= alpha_mask
            gt_image *= alpha_mask
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # Segmentation Loss
        loss_seg = torch.tensor(0.0, device="cuda")
        if opt.lambda_seg > 0 and viewpoint_cam.alpha_mask is not None:
            gt_seg = viewpoint_cam.alpha_mask.cuda()
            loss_seg_l1 = l1_loss(pred_seg, gt_seg)
            loss_seg_ssim = ssim(image, gt_image)
            loss_seg = (1.0 - opt.lambda_dssim) * loss_seg_l1 + opt.lambda_dssim * (1.0 - loss_seg_ssim)
            loss = loss + opt.lambda_seg * loss_seg

        # Depth Loss
        loss_depth = torch.tensor(0.0, device="cuda")
        if opt.lambda_depth > 0:
            gt_depth = viewpoint_cam.depth.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                loss_depth = depth_loss(depth, gt_depth, alpha_mask)
            else:
                loss_depth = depth_loss(depth, gt_depth)
            loss = loss + opt.lambda_depth * loss_depth

        # Normal Loss (rendered normals & normals estimated from omnidata)
        loss_normal = torch.tensor(0.0, device="cuda")
        if opt.lambda_normal > 0:
            gt_normal = viewpoint_cam.normal.cuda()
            if viewpoint_cam.alpha_mask is not None:
                alpha_mask = viewpoint_cam.alpha_mask.cuda()
                loss_normal = normal_loss(normal, gt_normal, alpha_mask)
            else:
                loss_normal = normal_loss(normal, gt_normal)
            loss = loss + opt.lambda_normal * loss_normal

        # Anisotropic Loss
        loss_anisotropic = torch.tensor(0.0, device="cuda")
        if opt.lambda_anisotropic > 0:
            loss_anisotropic = anisotropic_loss(gaussians.get_scaling)
            loss = loss + opt.lambda_anisotropic * loss_anisotropic

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss (no used)": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "L1 Loss": f"{Ll1.item():.{5}f}",
                                          "Depth Loss": f"{loss_depth.item():.{5}f}", "Normal Loss": f"{loss_normal.item():.{5}f}", 
                                          "Seg Loss": f"{loss_seg.item():.{5}f}", "Anisotropic Loss": f"{loss_anisotropic.item():.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1], use_gsplat=True)  # default using gsplat

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args: Namespace) -> SummaryWriter | None:
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer: SummaryWriter | None,
    iteration: int,
    Ll1: torch.Tensor,
    loss: torch.Tensor,
    l1_loss: torch.Tensor,
    elapsed: float,
    testing_iterations: list[int],
    scene: Scene,
    renderFunc,
    renderArgs,
    train_test_exp: bool,
) -> None:
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
```

inference_warp.py
```python
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"

    cfg.load_first_order_params(yaml_path, optimal_path)

    logger.info(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"


    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    assert len(glob.glob(f"{base_dir}/train/best_*.pth")) > 0
    best_model_path = glob.glob(f"{base_dir}/train/best_*.pth")[0]
    trainer.test(best_model_path)
```

interactive_playground.py
```python
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="./data/bg.png",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    parser.add_argument(
        "--inv_ctrl", action="store_true", help="invert horizontal control direction"
    )
    opt_group = parser.add_mutually_exclusive_group()
    opt_group.add_argument(
        "--use_optimal",
        dest="use_optimal",
        action="store_true",
        help="(default) load optimal_params.pkl",
    )
    opt_group.add_argument(
        "--no_use_optimal",
        dest="use_optimal",
        action="store_false",
        help="run from YAML only",
    )
    parser.add_argument(
        "--ignore_checkpoint_stiffness",
        action="store_true",
        help="do not load spring stiffness from the trained checkpoint",
    )
    parser.set_defaults(use_optimal=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    optimal_path = f"./experiments_optimization/{args.case_name}/optimal_params.pkl"
    logger.info(f"Loading optimal parameters from: {optimal_path}")
    if not os.path.exists(optimal_path):
        raise FileNotFoundError(
            f"{args.case_name}: Optimal parameters not found at {optimal_path}"
        )

    cfg.load_first_order_params(
        yaml_path, optimal_path, use_global_spring_Y=args.use_optimal
    )
    base_dir = f"./temp_experiments/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.bg_img_path = args.bg_img_path

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.interactive_playground(
        best_model_path,
        gaussians_path,
        args.n_ctrl_parts,
        args.inv_ctrl,
        ignore_checkpoint_stiffness=args.ignore_checkpoint_stiffness,
    )
```

optimize_cma.py
```python
# The first stage to optimize the sparse parameters using CMA-ES
from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import torch
import pickle
import json
from argparse import ArgumentParser


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter

    if "cloth" in case_name or "package" in case_name:
        cfg.load_zero_order_params("configs/cloth.yaml")
    else:
        cfg.load_zero_order_params("configs/real.yaml")

    base_dir = f"experiments_optimization/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    optimizer.optimize(max_iter=max_iter)
```

process_data.py
```python
import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
args = parser.parse_args()

# Set the debug flags
PROCESS_SEG = True
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
TEXT_PROMPT = f"{category}.hand"
CONTROLLER_NAME = "hand"
SHAPE_PRIOR = args.shape_prior

logger = None


def setup_logger(log_file: str = "timer.log") -> None:
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, task_name: str) -> None:
        self.task_name = task_name

    def __enter__(self) -> None:
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb,
    ) -> None:
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SEG:
    # Get the masks of the controller and the object using GroundedSAM2
    with Timer("Video Segmentation"):
        os.system(
            f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT {TEXT_PROMPT}"
        )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    # Get the mask path for the image
    with open(f"{base_path}/{case_name}/mask/mask_info_{0}.json", "r") as f:
        data = json.load(f)
    obj_idx = None
    for key, value in data.items():
        if value != CONTROLLER_NAME:
            if obj_idx is not None:
                raise ValueError("More than one object detected.")
            obj_idx = int(key)
    mask_path = f"{base_path}/{case_name}/mask/0/{obj_idx}/0.png"

    existDir(f"{base_path}/{case_name}/shape")
    # Get the high-resolution of the image to prepare for the trellis generation
    with Timer("Image Upscale"):
        if not os.path.isfile(f"{base_path}/{case_name}/shape/high_resolution.png"):
            os.system(
                f"python ./data_process/image_upscale.py --img_path {base_path}/{case_name}/color/0/0.png --mask_path {mask_path} --output_path {base_path}/{case_name}/shape/high_resolution.png --category {category}"
            )

    # Get the masked image of the object
    with Timer("Image Segmentation"):
        os.system(
            f"python ./data_process/segment_util_image.py --img_path {base_path}/{case_name}/shape/high_resolution.png --TEXT_PROMPT {category} --output_path {base_path}/{case_name}/shape/masked_image.png"
        )

    with Timer("Shape Prior Generation"):
        os.system(
            f"python ./data_process/shape_prior.py --img_path {base_path}/{case_name}/shape/masked_image.png --output_dir {base_path}/{case_name}/shape"
        )

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        os.system(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        os.system(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        os.system(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        os.system(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Align the shape prior with partial observation
    with Timer("Alignment"):
        os.system(
            f"python ./data_process/align.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )
        else:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
```

qqtt/__init__.py
```python
from .model import SpringMassSystemWarp
from .engine import InvPhyTrainerWarp, OptimizerCMA
```

qqtt/data/__init__.py
```python
from .simple_data import SimpleData
from .real_data import RealData```

qqtt/data/real_data.py
```python
import numpy as np
import torch
import pickle
from qqtt.utils import logger, visualize_pc, cfg
import matplotlib.pyplot as plt


class RealData:
    def __init__(self, visualize=False, save_gt=True):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")
        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        object_points = data["object_points"]
        object_colors = data["object_colors"]
        object_visibilities = data["object_visibilities"]
        object_motions_valid = data["object_motions_valid"]
        controller_points = data["controller_points"]
        other_surface_points = data["surface_points"]
        interior_points = data["interior_points"]

        # Get the rainbow color for the object_colors
        y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
        y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
        rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

        self.num_original_points = object_points.shape[1]
        self.num_surface_points = (
            self.num_original_points + other_surface_points.shape[0]
        )
        self.num_all_points = self.num_surface_points + interior_points.shape[0]

        # Concatenate the surface points and interior points
        self.structure_points = np.concatenate(
            [object_points[0], other_surface_points, interior_points], axis=0
        )
        self.structure_points = torch.tensor(
            self.structure_points, dtype=torch.float32, device=cfg.device
        )

        self.object_points = torch.tensor(
            object_points, dtype=torch.float32, device=cfg.device
        )
        # self.object_colors = torch.tensor(
        #     object_colors, dtype=torch.float32, device=cfg.device
        # )
        self.original_object_colors = torch.tensor(
            object_colors, dtype=torch.float32, device=cfg.device
        )
        # Apply the rainbow color to the object_colors
        rainbow_colors = torch.tensor(
            rainbow_colors, dtype=torch.float32, device=cfg.device
        )
        # Make the same rainbow color for each frame
        self.object_colors = rainbow_colors.repeat(self.object_points.shape[0], 1, 1)

        # # Apply the first frame color to all frames
        # first_frame_colors = torch.tensor(
        #     object_colors[0], dtype=torch.float32, device=cfg.device
        # )
        # self.object_colors = first_frame_colors.repeat(self.object_points.shape[0], 1, 1)

        self.object_visibilities = torch.tensor(
            object_visibilities, dtype=torch.bool, device=cfg.device
        )
        self.object_motions_valid = torch.tensor(
            object_motions_valid, dtype=torch.bool, device=cfg.device
        )
        self.controller_points = torch.tensor(
            controller_points, dtype=torch.float32, device=cfg.device
        )

        self.frame_len = self.object_points.shape[0]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize, save_gt=save_gt)

    def visualize_data(self, visualize=False, save_gt=True):
        if visualize:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=True,
            )
        if save_gt:
            visualize_pc(
                self.object_points,
                self.object_colors,
                self.controller_points,
                self.object_visibilities,
                self.object_motions_valid,
                visualize=False,
                save_video=True,
                save_path=f"{self.base_dir}/gt.mp4",
            )
```

qqtt/data/simple_data.py
```python
# The simplest test data with full 3D point trajectories (n_frames, n_points, 3)
import numpy as np
import torch
from qqtt.utils import logger, visualize_pc, cfg


class SimpleData:
    def __init__(self, visualize=False):
        logger.info(f"[DATA]: loading data from {cfg.data_path}")

        self.data_path = cfg.data_path
        self.base_dir = cfg.base_dir
        self.data = np.load(self.data_path)
        self.data = torch.tensor(self.data, dtype=torch.float32, device=cfg.device)
        self.frame_len = self.data.shape[0]
        self.point_num = self.data.shape[1]
        # Visualize/save the GT frames
        self.visualize_data(visualize=visualize)

    def visualize_data(self, visualize=False):
        if visualize:
            visualize_pc(
                self.data,
                visualize=True,
            )
        visualize_pc(
            self.data,
            visualize=False,
            save_video=True,
            save_path=f"{self.base_dir}/gt.mp4",
        )
```

qqtt/engine/__init__.py
```python
from .cma_optimize_warp import OptimizerCMA
from .trainer_warp import InvPhyTrainerWarp```

qqtt/engine/cma_optimize_warp.py
```python
from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import SpringMassSystemWarp
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm
import warp as wp
import cma
import pickle
import os


class OptimizerCMA:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame,
        mask_path=None,
        velocity_path=None,
        device="cuda:0",
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        if not os.path.exists(f"{cfg.base_dir}/optimizeCMA"):
            # Create directory if it doesn't exist
            os.makedirs(f"{cfg.base_dir}/optimizeCMA")

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def normalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return (value - min) / (max - min)

    def denormalize(self, value, min, max):
        assert min < max, "The minimum value should be less than the maximum value"
        return value * (max - min) + min

    def optimize(self, max_iter=100):
        # Initialize the parameters
        init_global_spring_Y = self.normalize(
            cfg.init_spring_Y, cfg.spring_Y_min, cfg.spring_Y_max
        )
        init_object_radius = self.normalize(cfg.object_radius, 0.01, 0.05)
        init_object_max_neighbours = self.normalize(cfg.object_max_neighbours, 10, 50)
        init_controller_radius = self.normalize(cfg.controller_radius, 0.01, 0.08)
        init_controller_max_neighbours = self.normalize(
            cfg.controller_max_neighbours, 10, 80
        )
        init_collide_elas = cfg.collide_elas
        init_collide_fric = self.normalize(cfg.collide_fric, 0, 2)
        init_collide_object_elas = cfg.collide_object_elas
        init_collide_object_fric = self.normalize(cfg.collide_object_fric, 0, 2)
        init_collision_dist = self.normalize(cfg.collision_dist, 0.01, 0.05)
        init_drag_damping = self.normalize(cfg.drag_damping, 0, 20)
        init_dashpot_damping = self.normalize(cfg.dashpot_damping, 0, 200)

        x_init = [
            init_global_spring_Y,
            init_object_radius,
            init_object_max_neighbours,
            init_controller_radius,
            init_controller_max_neighbours,
            init_collide_elas,
            init_collide_fric,
            init_collide_object_elas,
            init_collide_object_fric,
            init_collision_dist,
            init_drag_damping,
            init_dashpot_damping,
        ]

        self.error_func(
            x_init, visualize=True, video_path=f"{cfg.base_dir}/optimizeCMA/init.mp4"
        )

        std = 1 / 6
        es = cma.CMAEvolutionStrategy(x_init, std, {"bounds": [0.0, 1.0], "seed": 42})
        es.optimize(self.error_func, iterations=max_iter)

        # Get the results
        res = es.result
        optimal_x = np.array(res[0]).astype(np.float32)
        optimal_error = res[1]
        logger.info(f"Optimal x: {optimal_x}, Optimal error: {optimal_error}")

        final_global_spring_Y = self.denormalize(
            optimal_x[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        final_object_radius = self.denormalize(optimal_x[1], 0.01, 0.05)
        final_object_max_neighbours = int(self.denormalize(optimal_x[2], 10, 50))
        final_controller_radius = self.denormalize(optimal_x[3], 0.01, 0.08)
        final_controller_max_neighbours = int(self.denormalize(optimal_x[4], 10, 80))
        final_collide_elas = optimal_x[5]
        final_collide_fric = self.denormalize(optimal_x[6], 0, 2)
        final_collide_object_elas = optimal_x[7]
        final_collide_object_fric = self.denormalize(optimal_x[8], 0, 2)
        final_collision_dist = self.denormalize(optimal_x[9], 0.01, 0.05)
        final_drag_damping = self.denormalize(optimal_x[10], 0, 20)
        final_dashpot_damping = self.denormalize(optimal_x[11], 0, 200)

        self.error_func(
            optimal_x,
            visualize=True,
            video_path=f"{cfg.base_dir}/optimizeCMA/optimal.mp4",
        )

        optimal_results = {}
        optimal_results["global_spring_Y"] = final_global_spring_Y
        optimal_results["object_radius"] = final_object_radius
        optimal_results["object_max_neighbours"] = final_object_max_neighbours
        optimal_results["controller_radius"] = final_controller_radius
        optimal_results["controller_max_neighbours"] = final_controller_max_neighbours
        optimal_results["collide_elas"] = final_collide_elas
        optimal_results["collide_fric"] = final_collide_fric
        optimal_results["collide_object_elas"] = final_collide_object_elas
        optimal_results["collide_object_fric"] = final_collide_object_fric
        optimal_results["collision_dist"] = final_collision_dist
        optimal_results["drag_damping"] = final_drag_damping
        optimal_results["dashpot_damping"] = final_dashpot_damping

        # Save out all the initialized parameters
        with open(f"{cfg.base_dir}/optimal_params.pkl", "wb") as f:
            pickle.dump(optimal_results, f)

    def error_func(self, parameters, visualize=False, video_path=None):
        global_spring_Y = self.denormalize(
            parameters[0], cfg.spring_Y_min, cfg.spring_Y_max
        )
        object_radius = self.denormalize(parameters[1], 0.01, 0.05)
        object_max_neighbours = int(self.denormalize(parameters[2], 10, 50))
        controller_radius = self.denormalize(parameters[3], 0.01, 0.08)
        controller_max_neighbours = int(self.denormalize(parameters[4], 10, 80))
        collide_elas = parameters[5]
        collide_fric = self.denormalize(parameters[6], 0, 2)
        collide_object_elas = parameters[7]
        collide_object_fric = self.denormalize(parameters[8], 0, 2)
        collision_dist = self.denormalize(parameters[9], 0.01, 0.05)
        drag_damping = self.denormalize(parameters[10], 0, 20)
        dashpot_damping = self.denormalize(parameters[11], 0, 200)

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=object_radius,
            object_max_neighbours=object_max_neighbours,
            controller_radius=controller_radius,
            controller_max_neighbours=controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=global_spring_Y,
            collide_elas=collide_elas,
            collide_fric=collide_fric,
            dashpot_damping=dashpot_damping,
            drag_damping=drag_damping,
            collide_object_elas=collide_object_elas,
            collide_object_fric=collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
            disable_backward=True,
        )

        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )

        if visualize == True:
            vertices = [
                wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
            ]

        if cfg.data_type == "real":
            self.simulator.set_acc_count(False)

        total_loss = 0.0
        if not visualize:
            # Only optimize on the train frames
            max_frame = cfg.train_frame
        else:
            max_frame = self.dataset.frame_len

        for j in range(1, max_frame):
            self.simulator.set_controller_target(j)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            if cfg.use_graph:
                wp.capture_launch(self.simulator.graph)
            else:
                if cfg.data_type == "real":
                    with self.simulator.tape:
                        self.simulator.step()
                        self.simulator.calculate_loss()
                else:
                    with self.simulator.tape:
                        self.simulator.step()
                        self.simulator.calculate_simple_loss()

            if visualize == True:
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())

            if cfg.data_type == "real":
                if wp.to_torch(self.simulator.acc_count, requires_grad=False)[0] == 0:
                    self.simulator.set_acc_count(True)

                # Update the prev_acc used to calculate the acceleration loss
                self.simulator.update_acc()

            loss = wp.to_torch(self.simulator.loss, requires_grad=False)
            total_loss += loss.item()

            self.simulator.clear_loss()
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

        total_loss /= cfg.train_frame - 1

        if visualize == True:
            vertices = torch.stack(vertices, dim=0)
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

        return total_loss```

qqtt/engine/trainer_warp.py
```python
from qqtt.data import RealData, SimpleData
from qqtt.utils import logger, visualize_pc, cfg
from qqtt.model.diff_simulator import (
    SpringMassSystemWarp,
)
import open3d as o3d
import numpy as np
import torch
import wandb
import os
from tqdm import tqdm
import warp as wp
from scipy.spatial import KDTree
import pickle
import cv2
from pynput import keyboard
import pyrender
import trimesh
import matplotlib.pyplot as plt

from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)
from gaussian_splatting.utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from gs_render import (
    remove_gaussians_with_low_opacity,
    remove_gaussians_with_point_mesh_distance,
)
from gaussian_splatting.rotation_utils import quaternion_multiply, matrix_to_quaternion

from sklearn.cluster import KMeans
import copy
import time
import threading
import time


class InvPhyTrainerWarp:
    def __init__(
        self,
        data_path,
        base_dir,
        train_frame=None,
        mask_path=None,
        velocity_path=None,
        pure_inference_mode=False,
        device="cuda:0",
    ):
        cfg.data_path = data_path
        cfg.base_dir = base_dir
        cfg.device = device
        cfg.run_name = base_dir.split("/")[-1]
        cfg.train_frame = train_frame

        self.init_masks = None
        self.init_velocities = None
        # Load the data
        if cfg.data_type == "real":
            self.dataset = RealData(visualize=False, save_gt=False)
            # Get the object points and controller points
            self.object_points = self.dataset.object_points
            self.object_colors = self.dataset.object_colors
            self.object_visibilities = self.dataset.object_visibilities
            self.object_motions_valid = self.dataset.object_motions_valid
            self.controller_points = self.dataset.controller_points
            self.structure_points = self.dataset.structure_points
            self.num_original_points = self.dataset.num_original_points
            self.num_surface_points = self.dataset.num_surface_points
            self.num_all_points = self.dataset.num_all_points
        elif cfg.data_type == "synthetic":
            self.dataset = SimpleData(visualize=False)
            self.object_points = self.dataset.data
            self.object_colors = None
            self.object_visibilities = None
            self.object_motions_valid = None
            self.controller_points = None
            self.structure_points = self.dataset.data[0]
            self.num_original_points = None
            self.num_surface_points = None
            self.num_all_points = len(self.dataset.data[0])
            # Prepare for the multiple object case
            if mask_path is not None:
                mask = np.load(mask_path)
                self.init_masks = torch.tensor(
                    mask, dtype=torch.float32, device=cfg.device
                )
            if velocity_path is not None:
                velocity = np.load(velocity_path)
                self.init_velocities = torch.tensor(
                    velocity, dtype=torch.float32, device=cfg.device
                )
        else:
            raise ValueError(f"Data type {cfg.data_type} not supported")

        # Initialize the vertices, springs, rest lengths and masses
        if self.controller_points is None:
            firt_frame_controller_points = None
        else:
            firt_frame_controller_points = self.controller_points[0]
        (
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            self.num_object_springs,
        ) = self._init_start(
            self.structure_points,
            firt_frame_controller_points,
            object_radius=cfg.object_radius,
            object_max_neighbours=cfg.object_max_neighbours,
            controller_radius=cfg.controller_radius,
            controller_max_neighbours=cfg.controller_max_neighbours,
            mask=self.init_masks,
        )

        self.simulator = SpringMassSystemWarp(
            self.init_vertices,
            self.init_springs,
            self.init_rest_lengths,
            self.init_masses,
            dt=cfg.dt,
            num_substeps=cfg.num_substeps,
            spring_Y=cfg.init_spring_Y,
            collide_elas=cfg.collide_elas,
            collide_fric=cfg.collide_fric,
            dashpot_damping=cfg.dashpot_damping,
            drag_damping=cfg.drag_damping,
            collide_object_elas=cfg.collide_object_elas,
            collide_object_fric=cfg.collide_object_fric,
            init_masks=self.init_masks,
            collision_dist=cfg.collision_dist,
            init_velocities=self.init_velocities,
            num_object_points=self.num_all_points,
            num_surface_points=self.num_surface_points,
            num_original_points=self.num_original_points,
            controller_points=self.controller_points,
            reverse_z=cfg.reverse_z,
            spring_Y_min=cfg.spring_Y_min,
            spring_Y_max=cfg.spring_Y_max,
            gt_object_points=self.object_points,
            gt_object_visibilities=self.object_visibilities,
            gt_object_motions_valid=self.object_motions_valid,
            self_collision=cfg.self_collision,
        )

        if not pure_inference_mode:
            self.optimizer = torch.optim.Adam(
                [
                    wp.to_torch(self.simulator.wp_spring_Y),
                    wp.to_torch(self.simulator.wp_collide_elas),
                    wp.to_torch(self.simulator.wp_collide_fric),
                    wp.to_torch(self.simulator.wp_collide_object_elas),
                    wp.to_torch(self.simulator.wp_collide_object_fric),
                ],
                lr=cfg.base_lr,
                betas=(0.9, 0.99),
            )

            if "debug" not in cfg.run_name:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="final_pipeline",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            else:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="Debug",
                    name=cfg.run_name,
                    config=cfg.to_dict(),
                )
            if not os.path.exists(f"{cfg.base_dir}/train"):
                # Create directory if it doesn't exist
                os.makedirs(f"{cfg.base_dir}/train")

    def _init_start(
        self,
        object_points,
        controller_points,
        object_radius=0.02,
        object_max_neighbours=30,
        controller_radius=0.04,
        controller_max_neighbours=50,
        mask=None,
    ):
        object_points = object_points.cpu().numpy()
        if controller_points is not None:
            controller_points = controller_points.cpu().numpy()
        if mask is None:
            object_pcd = o3d.geometry.PointCloud()
            object_pcd.points = o3d.utility.Vector3dVector(object_points)
            pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

            # Connect the springs of the objects first
            points = np.asarray(object_pcd.points)
            spring_flags = np.zeros((len(points), len(points)))
            springs = []
            rest_lengths = []
            for i in range(len(points)):
                [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                    points[i], object_radius, object_max_neighbours
                )
                idx = idx[1:]
                for j in idx:
                    rest_length = np.linalg.norm(points[i] - points[j])
                    if (
                        spring_flags[i, j] == 0
                        and spring_flags[j, i] == 0
                        and rest_length > 1e-4
                    ):
                        spring_flags[i, j] = 1
                        spring_flags[j, i] = 1
                        springs.append([i, j])
                        rest_lengths.append(np.linalg.norm(points[i] - points[j]))

            num_object_springs = len(springs)

            if controller_points is not None:
                # Connect the springs between the controller points and the object points
                num_object_points = len(points)
                points = np.concatenate([points, controller_points], axis=0)
                for i in range(len(controller_points)):
                    [k, idx, _] = pcd_tree.search_hybrid_vector_3d(
                        controller_points[i],
                        controller_radius,
                        controller_max_neighbours,
                    )
                    for j in idx:
                        springs.append([num_object_points + i, j])
                        rest_lengths.append(
                            np.linalg.norm(controller_points[i] - points[j])
                        )

            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(points))
            return (
                torch.tensor(points, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )
        else:
            mask = mask.cpu().numpy()
            # Get the unique value in masks
            unique_values = np.unique(mask)
            vertices = []
            springs = []
            rest_lengths = []
            index = 0
            # Loop different objects to connect the springs separately
            for value in unique_values:
                temp_points = object_points[mask == value]
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(temp_points)
                temp_tree = o3d.geometry.KDTreeFlann(temp_pcd)
                temp_spring_flags = np.zeros((len(temp_points), len(temp_points)))
                temp_springs = []
                temp_rest_lengths = []
                for i in range(len(temp_points)):
                    [k, idx, _] = temp_tree.search_hybrid_vector_3d(
                        temp_points[i], object_radius, object_max_neighbours
                    )
                    idx = idx[1:]
                    for j in idx:
                        rest_length = np.linalg.norm(temp_points[i] - temp_points[j])
                        if (
                            temp_spring_flags[i, j] == 0
                            and temp_spring_flags[j, i] == 0
                            and rest_length > 1e-4
                        ):
                            temp_spring_flags[i, j] = 1
                            temp_spring_flags[j, i] = 1
                            temp_springs.append([i + index, j + index])
                            temp_rest_lengths.append(rest_length)
                vertices += temp_points.tolist()
                springs += temp_springs
                rest_lengths += temp_rest_lengths
                index += len(temp_points)

            num_object_springs = len(springs)

            vertices = np.array(vertices)
            springs = np.array(springs)
            rest_lengths = np.array(rest_lengths)
            masses = np.ones(len(vertices))

            return (
                torch.tensor(vertices, dtype=torch.float32, device=cfg.device),
                torch.tensor(springs, dtype=torch.int32, device=cfg.device),
                torch.tensor(rest_lengths, dtype=torch.float32, device=cfg.device),
                torch.tensor(masses, dtype=torch.float32, device=cfg.device),
                num_object_springs,
            )

    def train(self, start_epoch=-1):
        # Render the initial visualization
        video_path = f"{cfg.base_dir}/train/init.mp4"
        self.visualize_sim(save_only=True, video_path=video_path)

        best_loss = None
        best_epoch = None
        # Train the model with the physical simulator
        for i in range(start_epoch + 1, cfg.iterations):
            total_loss = 0.0
            if cfg.data_type == "real":
                total_chamfer_loss = 0.0
                total_track_loss = 0.0
            self.simulator.set_init_state(
                self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
            )
            with wp.ScopedTimer("backward"):
                for j in tqdm(range(1, cfg.train_frame)):
                    self.simulator.set_controller_target(j)
                    if self.simulator.object_collision_flag:
                        self.simulator.update_collision_graph()

                    if cfg.use_graph:
                        wp.capture_launch(self.simulator.graph)
                    else:
                        if cfg.data_type == "real":
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_loss()
                            self.simulator.tape.backward(self.simulator.loss)
                        else:
                            with self.simulator.tape:
                                self.simulator.step()
                                self.simulator.calculate_simple_loss()
                            self.simulator.tape.backward(self.simulator.loss)

                    self.optimizer.step()

                    if cfg.data_type == "real":
                        chamfer_loss = wp.to_torch(
                            self.simulator.chamfer_loss, requires_grad=False
                        )
                        track_loss = wp.to_torch(
                            self.simulator.track_loss, requires_grad=False
                        )
                        total_chamfer_loss += chamfer_loss.item()
                        total_track_loss += track_loss.item()

                    loss = wp.to_torch(self.simulator.loss, requires_grad=False)
                    total_loss += loss.item()

                    if cfg.use_graph:
                        # Only need to clear the gradient, the tape is created in the graph
                        self.simulator.tape.zero()
                    else:
                        # Need to reset the compute graph and clear the gradient
                        self.simulator.tape.reset()
                    self.simulator.clear_loss()
                    # Set the intial state for the next step
                    self.simulator.set_init_state(
                        self.simulator.wp_states[-1].wp_x,
                        self.simulator.wp_states[-1].wp_v,
                    )

            total_loss /= cfg.train_frame - 1
            if cfg.data_type == "real":
                total_chamfer_loss /= cfg.train_frame - 1
                total_track_loss /= cfg.train_frame - 1
            wandb.log(
                {
                    "loss": total_loss,
                    "chamfer_loss": (
                        total_chamfer_loss if cfg.data_type == "real" else 0
                    ),
                    "track_loss": total_track_loss if cfg.data_type == "real" else 0,
                    "collide_else": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ).item(),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ).item(),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ).item(),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ).item(),
                },
                step=i,
            )

            logger.info(f"[Train]: Iteration: {i}, Loss: {total_loss}")

            if i % cfg.vis_interval == 0 or i == cfg.iterations - 1:
                video_path = f"{cfg.base_dir}/train/sim_iter{i}.mp4"
                self.visualize_sim(save_only=True, video_path=video_path)
                wandb.log(
                    {
                        "video": wandb.Video(
                            video_path,
                            format="mp4",
                            fps=cfg.FPS,
                        ),
                    },
                    step=i,
                )
                # Save the parameters
                cur_model = {
                    "epoch": i,
                    "num_object_springs": self.num_object_springs,
                    "spring_Y": torch.exp(
                        wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
                    ),
                    "collide_elas": wp.to_torch(
                        self.simulator.wp_collide_elas, requires_grad=False
                    ),
                    "collide_fric": wp.to_torch(
                        self.simulator.wp_collide_fric, requires_grad=False
                    ),
                    "collide_object_elas": wp.to_torch(
                        self.simulator.wp_collide_object_elas, requires_grad=False
                    ),
                    "collide_object_fric": wp.to_torch(
                        self.simulator.wp_collide_object_fric, requires_grad=False
                    ),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }
                if best_loss == None or total_loss < best_loss:
                    # Remove old best model file if it exists
                    if best_loss is not None:
                        old_best_model_path = (
                            f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                        )
                        if os.path.exists(old_best_model_path):
                            os.remove(old_best_model_path)

                    # Update best loss and best epoch
                    best_loss = total_loss
                    best_epoch = i

                    # Save new best model
                    best_model_path = f"{cfg.base_dir}/train/best_{best_epoch}.pth"
                    torch.save(cur_model, best_model_path)
                    logger.info(
                        f"Latest best model saved: epoch {best_epoch} with loss {best_loss}"
                    )

                torch.save(cur_model, f"{cfg.base_dir}/train/iter_{i}.pth")
                logger.info(
                    f"[Visualize]: Visualize the simulation at iteration {i} and save the model"
                )

        wandb.finish()

    def test(self, model_path=None):
        if model_path is not None:
            # Load the model
            logger.info(f"Load model from {model_path}")
            checkpoint = torch.load(model_path, map_location=cfg.device)

            spring_Y = checkpoint["spring_Y"]
            collide_elas = checkpoint["collide_elas"]
            collide_fric = checkpoint["collide_fric"]
            collide_object_elas = checkpoint["collide_object_elas"]
            collide_object_fric = checkpoint["collide_object_fric"]
            num_object_springs = checkpoint["num_object_springs"]

            assert (
                len(spring_Y) == self.simulator.n_springs
            ), "Check if the loaded checkpoint match the config file to connect the springs"

            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
            self.simulator.set_collide(
                collide_elas.detach().clone(), collide_fric.detach().clone()
            )
            self.simulator.set_collide_object(
                collide_object_elas.detach().clone(),
                collide_object_fric.detach().clone(),
            )

        # Render the initial visualization
        video_path = f"{cfg.base_dir}/inference.mp4"
        save_path = f"{cfg.base_dir}/inference.pkl"
        self.visualize_sim(
            save_only=True,
            video_path=video_path,
            save_trajectory=True,
            save_path=save_path,
        )

    def visualize_sim(
        self, save_only=True, video_path=None, save_trajectory=False, save_path=None
    ):
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        vertices = [
            wp.to_torch(self.simulator.wp_states[0].wp_x, requires_grad=False).cpu()
        ]

        with wp.ScopedTimer("simulate"):
            for i in tqdm(range(1, frame_len)):
                if cfg.data_type == "real":
                    self.simulator.set_controller_target(i, pure_inference=True)
                if self.simulator.object_collision_flag:
                    self.simulator.update_collision_graph()

                if cfg.use_graph:
                    wp.capture_launch(self.simulator.forward_graph)
                else:
                    self.simulator.step()
                x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
                vertices.append(x.cpu())
                # Set the intial state for the next step
                self.simulator.set_init_state(
                    self.simulator.wp_states[-1].wp_x,
                    self.simulator.wp_states[-1].wp_v,
                )

        vertices = torch.stack(vertices, dim=0)

        if save_trajectory:
            logger.info(f"Save the trajectory to {save_path}")
            vertices_to_save = vertices.cpu().numpy()
            with open(save_path, "wb") as f:
                pickle.dump(vertices_to_save, f)

        if not save_only:
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=True,
            )
        else:
            assert video_path is not None, "Please provide the video path to save"
            visualize_pc(
                vertices[:, : self.num_all_points, :],
                self.object_colors,
                self.controller_points,
                visualize=False,
                save_video=True,
                save_path=video_path,
            )

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        for key in self.pressed_keys:
            if key in self.key_mappings:
                idx, change = self.key_mappings[key]
                target_change[idx] += change
        return target_change

    def init_control_ui(self):

        height = cfg.WH[1]
        width = cfg.WH[0]

        self.arrow_size = 30

        self.arrow_empty_orig = cv2.imread(
            "./assets/arrow_empty.png", cv2.IMREAD_UNCHANGED
        )[:, :, [2, 1, 0, 3]]
        self.arrow_1_orig = cv2.imread("./assets/arrow_1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.arrow_2_orig = cv2.imread("./assets/arrow_2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        spacing = self.arrow_size + 5

        self.bottom_margin = 25  # Margin from bottom of screen
        bottom_y = height - self.bottom_margin
        top_y = height - self.bottom_margin - spacing

        self.edge_buffer = self.bottom_margin
        set1_margin_x = self.edge_buffer  # Add buffer from left edge
        set2_margin_x = width - self.edge_buffer

        self.arrow_positions_set1 = {
            "q": (set1_margin_x + spacing * 3, top_y),  # Up
            "w": (set1_margin_x + spacing, top_y),  # Forward
            "a": (set1_margin_x, bottom_y),  # Left
            "s": (set1_margin_x + spacing, bottom_y),  # Backward
            "d": (set1_margin_x + spacing * 2, bottom_y),  # Right
            "e": (set1_margin_x + spacing * 3, bottom_y),  # Down
        }

        self.arrow_positions_set2 = {
            "u": (set2_margin_x - spacing * 3, top_y),  # Up
            "i": (set2_margin_x - spacing * 1, top_y),  # Forward
            "j": (set2_margin_x - spacing * 2, bottom_y),  # Left
            "k": (set2_margin_x - spacing * 1, bottom_y),  # Backward
            "l": (set2_margin_x, bottom_y),  # Right
            "o": (set2_margin_x - spacing * 3, bottom_y),  # Down
        }

        self.interm_size = 512
        self.rotations = {
            "w": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "a": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "s": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "d": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "q": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "e": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
            "i": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Forward
            "j": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 90, 1
            ),  # Left
            "k": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Backward
            "l": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 270, 1
            ),  # Right
            "u": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 0, 1
            ),  # Up
            "o": cv2.getRotationMatrix2D(
                (self.interm_size // 2, self.interm_size // 2), 180, 1
            ),  # Down
        }

        self.hand_left = cv2.imread("./assets/Picture2.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]
        self.hand_right = cv2.imread("./assets/Picture1.png", cv2.IMREAD_UNCHANGED)[
            :, :, [2, 1, 0, 3]
        ]

        self.hand_left_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)
        self.hand_right_pos = torch.tensor([0.0, 0.0, 0.0], device=cfg.device)

        # pre-compute all rotated arrows to avoid aliasing
        self.arrow_rotated_filled = {}
        self.arrow_rotated_empty = {}
        for key in self.arrow_positions_set1:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_1_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
        for key in self.arrow_positions_set2:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_2_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(
                        self.arrow_empty_orig,
                        (self.interm_size, self.interm_size),
                        interpolation=cv2.INTER_AREA,
                    ),
                    key,
                ),
                (self.arrow_size, self.arrow_size),
                interpolation=cv2.INTER_AREA,
            )

    def _rotate_arrow(self, arrow, key):
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow,
            rotation_matrix,
            (self.interm_size, self.interm_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT,
        )
        return rotated

    def _overlay_arrow(self, background, arrow, position, key, filled=True):
        x, y = position

        if filled:
            rotated_arrow = self.arrow_rotated_filled[key].copy()
        else:
            rotated_arrow = self.arrow_rotated_empty[key].copy()

        h, w = rotated_arrow.shape[:2]

        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)

        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)

        roi = background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        arrow_roi = rotated_arrow[arrow_y : arrow_y + roi_h, arrow_x : arrow_x + roi_w]

        alpha = arrow_roi[:, :, 3] / 255.0

        for c in range(3):  # Apply for RGB channels
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha

        background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi

        return background

    def _overlay_hand_at_position(
        self, frame, target_points, x_axis, hand_size, hand_icon, align="center"
    ):
        result = frame.copy()

        mean_pos = target_points.cpu().numpy().mean(axis=0)

        pixel_mean = self.projection @ np.append(mean_pos, 1)
        pixel_mean = pixel_mean[:2] / pixel_mean[2]

        pos_1 = np.append(mean_pos + hand_size * x_axis, 1)
        pixel_1 = self.projection @ pos_1
        pixel_1 = pixel_1[:2] / pixel_1[2]

        pos_2 = np.append(mean_pos - hand_size * x_axis, 1)
        pixel_2 = self.projection @ pos_2
        pixel_2 = pixel_2[:2] / pixel_2[2]

        icon_size = int(np.linalg.norm(pixel_1[:2] - pixel_2[:2]) / 2)
        icon_size = max(1, min(icon_size, 100))

        resized_icon = cv2.resize(hand_icon, (icon_size, icon_size))
        h, w = resized_icon.shape[:2]
        x, y = int(pixel_mean[0]), int(pixel_mean[1])

        if align == "top-left":
            roi_x = int(max(0, x - w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "top-right":
            roi_x = int(max(0, x - w + w * 0.15))
            roi_y = int(max(0, y - h * 0.1))
        if align == "center":
            roi_x = int(max(0, x - w // 2))
            roi_y = int(max(0, y - h // 2))
        roi_w = min(w, result.shape[1] - roi_x)
        roi_h = min(h, result.shape[0] - roi_y)

        if roi_w <= 0 or roi_h <= 0:
            return result

        icon_x = max(0, w // 2 - x)
        icon_y = max(0, h // 2 - y)

        roi = result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        icon_roi = resized_icon[icon_y : icon_y + roi_h, icon_x : icon_x + roi_w]

        if icon_roi.size == 0 or roi.shape[:2] != icon_roi.shape[:2]:
            return result

        if icon_roi.shape[2] == 4:
            alpha = icon_roi[:, :, 3] / 255.0
            for c in range(3):
                roi[:, :, c] = roi[:, :, c] * (1 - alpha) + icon_roi[:, :, c] * alpha
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi
        else:
            result[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = icon_roi[:, :, :3]

        return result

    def _overlay_hand_icons(self, frame):
        if self.n_ctrl_parts not in [1, 2]:
            raise ValueError("Only support 1 or 2 control parts")

        result = frame.copy()

        c2w = np.linalg.inv(self.w2c)
        x_axis = c2w[:3, 0]
        self.projection = self.intrinsic @ self.w2c[:3, :]
        hand_size = 0.1  # size in physical space (in meters)

        if self.n_ctrl_parts == 1:
            current_target = self.hand_left_pos.unsqueeze(0)
            # align = 'top-right'
            align = "center"
            result = self._overlay_hand_at_position(
                result, current_target, x_axis, hand_size, self.hand_left, align
            )
        else:
            for i in range(2):
                current_target = (
                    self.hand_left_pos.unsqueeze(0)
                    if i == 0
                    else self.hand_right_pos.unsqueeze(0)
                )
                # align = 'top-right' if i == 0 else 'top-left'
                align = "center"
                hand_icon = self.hand_left if i == 0 else self.hand_right
                result = self._overlay_hand_at_position(
                    result, current_target, x_axis, hand_size, hand_icon, align
                )

        return result

    def update_frame(self, frame, pressed_keys):
        result = frame.copy()

        result = self._overlay_hand_icons(result)

        # overlay an transparent white mask on the bottom left and bottom right corners with width trans_width, and height trans_height
        trans_width = 160
        trans_height = 120
        overlay = result.copy()

        bottom_left_pt1 = (0, cfg.WH[1] - trans_height)
        bottom_left_pt2 = (trans_width, cfg.WH[1])
        cv2.rectangle(overlay, bottom_left_pt1, bottom_left_pt2, (255, 255, 255), -1)

        if self.n_ctrl_parts == 2:
            bottom_right_pt1 = (cfg.WH[0] - trans_width, cfg.WH[1] - trans_height)
            bottom_right_pt2 = (cfg.WH[0], cfg.WH[1])
            cv2.rectangle(
                overlay, bottom_right_pt1, bottom_right_pt2, (255, 255, 255), -1
            )

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # Draw all buttons for Set 1 (left side)
        for key, pos in self.arrow_positions_set1.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, None, pos, key, filled=True)
            else:
                result = self._overlay_arrow(result, None, pos, key, filled=False)

        # Draw all buttons for Set 2 (right side)
        if self.n_ctrl_parts == 2:
            for key, pos in self.arrow_positions_set2.items():
                if key in pressed_keys:
                    result = self._overlay_arrow(result, None, pos, key, filled=True)
                else:
                    result = self._overlay_arrow(result, None, pos, key, filled=False)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        control1_x = self.edge_buffer  # hard coded for now
        control2_x = cfg.WH[0] - self.edge_buffer - 113  # hard coded for now
        text_y = (
            cfg.WH[1] - self.arrow_size * 2 - self.bottom_margin - 10
        )  # hard coded for now
        cv2.putText(
            result,
            "Left Hand",
            (control1_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        if self.n_ctrl_parts == 2:
            cv2.putText(
                result,
                "Right Hand",
                (control2_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return result

    def _find_closest_point(self, target_points):
        """Find the closest structure point to any of the target points."""
        dist_matrix = torch.sum(
            (target_points.unsqueeze(1) - self.structure_points.unsqueeze(0)) ** 2,
            dim=2,
        )
        min_dist_per_ctrl_pts, min_indices = torch.min(dist_matrix, dim=1)
        min_idx = min_indices[torch.argmin(min_dist_per_ctrl_pts)]
        return self.structure_points[min_idx].unsqueeze(0)

    def interactive_playground(
        self,
        model_path,
        gs_path,
        n_ctrl_parts=1,
        inv_ctrl=False,
        *,
        ignore_checkpoint_stiffness=False,
    ):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        if not ignore_checkpoint_stiffness:
            self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())

        # Print the final spring stiffness used by the simulator for debugging
        loaded_stiffness = (
            wp.to_torch(self.simulator.wp_spring_Y, requires_grad=False)
            .exp()
            .cpu()
            .numpy()
        )
        logger.info(
            f"Final spring stiffness (first 10 values): {loaded_stiffness[:10]}"
        )
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        ###########################################################################

        logger.info("Party Time Start!!!!")
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        current_target = self.simulator.controller_points[0]
        prev_target = current_target

        vis_controller_points = current_target.cpu().numpy()

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None
        image_path = cfg.bg_img_path
        overlay = cv2.imread(image_path)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay = torch.tensor(overlay, dtype=torch.float32, device=cfg.device)

        if n_ctrl_parts > 1:
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(vis_controller_points)
            N = vis_controller_points.shape[0]
            masks_ctrl_pts = []
            for i in range(n_ctrl_parts):
                mask = cluster_labels == i
                masks_ctrl_pts.append(torch.from_numpy(mask))
            # project the center of the cluster to the object to the image space, those on the left will be mask 1
            center1 = np.mean(vis_controller_points[masks_ctrl_pts[0]], axis=0)
            center2 = np.mean(vis_controller_points[masks_ctrl_pts[1]], axis=0)
            center1 = np.concatenate([center1, [1]])
            center2 = np.concatenate([center2, [1]])
            proj_mat = intrinsic @ w2c[:3, :]
            center1 = proj_mat @ center1
            center2 = proj_mat @ center2
            center1 = center1 / center1[-1]
            center2 = center2 / center2[-1]
            if center1[0] > center2[0]:
                print("Switching the control parts")
                masks_ctrl_pts = [masks_ctrl_pts[1], masks_ctrl_pts[0]]
        else:
            masks_ctrl_pts = None
        self.n_ctrl_parts = n_ctrl_parts
        self.mask_ctrl_pts = masks_ctrl_pts
        self.scale_factors = 1.0
        assert n_ctrl_parts <= 2, "Only support 1 or 2 control parts"
        print("UI Controls:")
        print("- Set 1: WASD (XY movement), QE (Z movement)")
        print("- Set 2: IJKL (XY movement), UO (Z movement)")
        self.inv_ctrl = -1.0 if inv_ctrl else 1.0
        self.key_mappings = {
            # Set 1 controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Set 2 controls
            "i": (1, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "k": (1, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "j": (1, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "l": (1, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "o": (1, np.array([0, 0, 0.005])),
            "u": (1, np.array([0, 0, -0.005])),
        }
        self.pressed_keys = set()
        self.w2c = w2c
        self.intrinsic = intrinsic
        self.init_control_ui()
        if n_ctrl_parts > 1:
            hand_positions = []
            for i in range(2):
                target_points = torch.from_numpy(
                    vis_controller_points[self.mask_ctrl_pts[i]]
                ).to("cuda")
                hand_positions.append(self._find_closest_point(target_points))
            self.hand_left_pos, self.hand_right_pos = hand_positions
        else:
            target_points = torch.from_numpy(vis_controller_points).to("cuda")
            self.hand_left_pos = self._find_closest_point(target_points)

        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        self.target_change = np.zeros((n_ctrl_parts, 3))

        ############## Temporary timer ##############
        import time

        class Timer:
            def __init__(self, name):
                self.name = name
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None
                self.use_cuda = torch.cuda.is_available()

            def start(self):
                if self.use_cuda:
                    torch.cuda.synchronize()
                    self.cuda_start_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_end_event = torch.cuda.Event(enable_timing=True)
                    self.cuda_start_event.record()
                self.start_time = time.time()

            def stop(self):
                if self.use_cuda:
                    self.cuda_end_event.record()
                    torch.cuda.synchronize()
                    self.elapsed = (
                        self.cuda_start_event.elapsed_time(self.cuda_end_event) / 1000
                    )  # convert ms to seconds
                else:
                    self.elapsed = time.time() - self.start_time
                return self.elapsed

            def reset(self):
                self.elapsed = 0
                self.start_time = None
                self.cuda_start_event = None
                self.cuda_end_event = None

        sim_timer = Timer("Simulator")
        render_timer = Timer("Rendering")
        frame_timer = Timer("Frame Compositing")
        interp_timer = Timer("Full Motion Interpolation")
        total_timer = Timer("Total Loop")
        knn_weights_timer = Timer("KNN Weights")
        motion_interp_timer = Timer("Motion Interpolation")

        # Performance stats
        fps_history = []
        component_times = {
            "simulator": [],
            "rendering": [],
            "frame_compositing": [],
            "full_motion_interpolation": [],
            "total": [],
            "knn_weights": [],
            "motion_interp": [],
        }

        # Number of frames to average over for stats
        STATS_WINDOW = 10
        frame_count = 0

        ############## End Temporary timer ##############

        while True:

            total_timer.start()

            # 1. Simulator step

            sim_timer.start()

            self.simulator.set_controller_interactive(prev_target, current_target)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()
            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            sim_time = sim_timer.stop()
            component_times["simulator"].append(sim_time)

            torch.cuda.synchronize()

            # 2. Frame initialization and setup

            frame_timer.start()

            frame = overlay.clone()

            frame_setup_time = (
                frame_timer.stop()
            )  # We'll accumulate times for frame compositing

            torch.cuda.synchronize()

            # 3. Rendering
            render_timer.start()

            # render with gaussians and paste the image on top of the frame
            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach()

            render_time = render_timer.stop()
            component_times["rendering"].append(render_time)

            torch.cuda.synchronize()

            # Continue frame compositing
            frame_timer.start()

            image = image.clamp(0, 1)
            if use_white_background:
                image_mask = torch.logical_and(
                    (image != 1.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = torch.logical_and(
                    (image != 0.0).any(dim=2), image[:, :, 3] > 100 / 255
                )
            image[..., 3].masked_fill_(~image_mask, 0.0)

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.cpu().numpy()
            image_mask = image_mask.cpu().numpy()
            frame = frame.astype(np.uint8)

            frame = self.update_frame(frame, self.pressed_keys)

            # Add shadows
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[0, 0, -3]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.95).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[1, 0.5, -2]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.97).astype(np.uint8)
            final_shadow = get_simple_shadow(
                x, intrinsic, w2c, width, height, image_mask, light_point=[-3, -0.5, -5]
            )
            frame[final_shadow] = (frame[final_shadow] * 0.98).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)

            frame_comp_time = (
                frame_timer.stop() + frame_setup_time
            )  # Total frame compositing time
            component_times["frame_compositing"].append(frame_comp_time)

            torch.cuda.synchronize()

            if prev_x is not None:
                with torch.no_grad():

                    prev_particle_pos = prev_x
                    cur_particle_pos = x

                    if relations is None:
                        relations = get_topk_indices(
                            prev_x, K=16
                        )  # only computed in the first iteration

                    if weights is None:
                        weights, weights_indices = knn_weights_sparse(
                            prev_particle_pos, current_pos, K=16
                        )  # only computed in the first iteration

                    interp_timer.start()

                    weights = calc_weights_vals_from_indices(
                        prev_particle_pos, current_pos, weights_indices
                    )

                    current_pos, current_rot, _ = interpolate_motions_speedup(
                        bones=prev_particle_pos,
                        motions=cur_particle_pos - prev_particle_pos,
                        relations=relations,
                        weights=weights,
                        weights_indices=weights_indices,
                        xyz=current_pos,
                        quat=current_rot,
                    )

                    # update gaussians with the new positions and rotations
                    gaussians._xyz = current_pos
                    gaussians._rotation = current_rot

                interp_time = interp_timer.stop()
                component_times["full_motion_interpolation"].append(interp_time)

            torch.cuda.synchronize()

            prev_x = x.clone()

            prev_target = current_target
            target_change = self.get_target_change()
            if masks_ctrl_pts is not None:
                for i in range(n_ctrl_parts):
                    if masks_ctrl_pts[i].sum() > 0:
                        current_target[masks_ctrl_pts[i]] += torch.tensor(
                            target_change[i], dtype=torch.float32, device=cfg.device
                        )
                        if i == 0:
                            self.hand_left_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
                        if i == 1:
                            self.hand_right_pos += torch.tensor(
                                target_change[i], dtype=torch.float32, device=cfg.device
                            )
            else:
                current_target += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )
                self.hand_left_pos += torch.tensor(
                    target_change, dtype=torch.float32, device=cfg.device
                )

            ############### Temporary timer ###############
            # Total loop time
            total_time = total_timer.stop()
            component_times["total"].append(total_time)

            # Calculate FPS
            fps = 1.0 / total_time
            fps_history.append(fps)

            # Display performance stats periodically
            frame_count += 1
            if frame_count % 10 == 0:
                # Limit stats to last STATS_WINDOW frames
                if len(fps_history) > STATS_WINDOW:
                    fps_history = fps_history[-STATS_WINDOW:]
                    for key in component_times:
                        component_times[key] = component_times[key][-STATS_WINDOW:]

                avg_fps = np.mean(fps_history)
                print(
                    f"\n--- Performance Stats (avg over last {len(fps_history)} frames) ---"
                )
                print(f"FPS: {avg_fps:.2f}")

                # Calculate percentages for pie chart
                total_avg = np.mean(component_times["total"])
                print(f"Total Frame Time: {total_avg*1000:.2f} ms")

                # Display individual component times
                for key in [
                    "simulator",
                    "rendering",
                    "frame_compositing",
                    "full_motion_interpolation",
                    "knn_weights",
                    "motion_interp",
                ]:
                    avg_time = np.mean(component_times[key])
                    percentage = (avg_time / total_avg) * 100
                    print(
                        f"{key.capitalize()}: {avg_time*1000:.2f} ms ({percentage:.1f}%)"
                    )

        listener.stop()

    def _transform_gs(self, gaussians, M, majority_scale=1):

        new_gaussians = copy.copy(gaussians)

        new_xyz = gaussians.get_xyz.clone()
        ones = torch.ones(
            (new_xyz.shape[0], 1), device=new_xyz.device, dtype=new_xyz.dtype
        )
        new_xyz = torch.cat((new_xyz, ones), dim=1)
        print("inside:", new_xyz.max(), new_xyz.min())
        new_xyz = new_xyz @ M.T
        print("outside:", new_xyz.max(), new_xyz.min())

        new_rotation = gaussians.get_rotation.clone()
        new_rotation = quaternion_multiply(
            matrix_to_quaternion(M[:3, :3]), new_rotation
        )

        new_scales = gaussians._scaling.clone()
        new_scales += torch.log(
            torch.tensor(
                majority_scale, device=new_scales.device, dtype=new_scales.dtype
            )
        )

        new_gaussians._xyz = new_xyz[:, :3]
        new_gaussians._rotation = new_rotation
        new_gaussians._scaling = new_scales

        return new_gaussians

    def _create_gs_view(self, w2c, intrinsic, height, width):
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        K = torch.tensor(intrinsic, dtype=torch.float32, device="cuda")
        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        return view

    def visualize_force(self, model_path, gs_path, n_ctrl_parts=2, force_scale=30000):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/force_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Get the controller points index
        first_frame_controller_points = self.simulator.controller_points[0]
        force_indexes = []
        if n_ctrl_parts == 1:
            force_indexes.append(
                torch.arange(first_frame_controller_points.shape[0], device=cfg.device)
            )
        else:
            # Use kmeans to find the two set of controller points
            kmeans = KMeans(n_clusters=n_ctrl_parts, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(
                first_frame_controller_points.cpu().numpy()
            )
            for i in range(n_ctrl_parts):
                force_indexes.append(
                    torch.tensor(np.where(cluster_labels == i)[0], device=cfg.device)
                )

        # Preprocess to get all the springs for different set of control points
        control_springs = self.init_springs[num_object_springs:]

        # Judge the springs whose left point is in the force_indexes
        force_springs = []
        force_object_points = []
        force_rest_lengths = []
        force_spring_Y = []

        for i in range(n_ctrl_parts):
            force_springs.append([])
            force_rest_lengths.append([])
            force_spring_Y.append([])
            force_object_points.append([])
            for j in range(len(control_springs)):
                if (control_springs[j][0] - self.num_all_points) in force_indexes[i]:
                    force_springs[i].append(control_springs[j])
                    force_rest_lengths[i].append(
                        self.init_rest_lengths[j + num_object_springs]
                    )
                    force_spring_Y[i].append(spring_Y[j + num_object_springs])
                    force_object_points[i].append(control_springs[j][1])
            force_springs[i] = torch.vstack(force_springs[i])
            force_springs[i][:, 0] -= self.num_all_points
            force_rest_lengths[i] = torch.tensor(
                force_rest_lengths[i], device=cfg.device
            )
            force_spring_Y[i] = torch.tensor(force_spring_Y[i], device=cfg.device)
            force_object_points[i] = torch.tensor(
                force_object_points[i], device=cfg.device
            )

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        force_arrow_meshes = []
        for j in range(n_ctrl_parts):
            # Calculate the center of the force_object_points
            force_center = (
                torch.mean(prev_x[force_object_points[j]], dim=0).cpu().numpy()
            )
            # Calculate the force vector
            force_vector = (
                self.get_force_vector(
                    prev_x,
                    force_springs[j],
                    force_rest_lengths[j],
                    force_spring_Y[j],
                    self.num_all_points,
                    self.simulator.controller_points[0],
                )
                .cpu()
                .numpy()
            )
            # Create arrow mesh in open3d
            if not (force_vector == 0).all():
                arrow_mesh = getArrowMesh(
                    origin=force_center,
                    end=force_center + force_vector / force_scale,
                    color=[1, 0, 0],
                )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])
        # Adjust the viewpoint
        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        force_image = (force_image * 255).astype(np.uint8)
        force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
        frame[~force_vis_mask] = force_image[~force_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imshow("Interactive Playground", frame)
        # cv2.waitKey(0)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            for arrow_mesh in force_arrow_meshes:
                vis.remove_geometry(arrow_mesh)

            force_arrow_meshes = []
            for j in range(n_ctrl_parts):
                # Calculate the center of the force_object_points
                force_center = (
                    torch.mean(x[force_object_points[j]], dim=0).cpu().numpy()
                )
                # Calculate the force vector
                force_vector = (
                    self.get_force_vector(
                        x,
                        force_springs[j],
                        force_rest_lengths[j],
                        force_spring_Y[j],
                        self.num_all_points,
                        self.simulator.controller_points[i],
                    )
                    .cpu()
                    .numpy()
                )
                if not (force_vector == 0).all():
                    # Create arrow mesh in open3d
                    arrow_mesh = getArrowMesh(
                        origin=force_center,
                        end=force_center + force_vector / force_scale,
                        color=[1, 0, 0],
                    )
                force_arrow_meshes.append(arrow_mesh)
                vis.add_geometry(force_arrow_meshes[j])

            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            # cv2.imshow("Interactive Playground", frame)
            # cv2.waitKey(0)
        vis.destroy_window()
        video_writer.release()

    def get_force_vector(
        self, x, springs, rest_lengths, spring_Y, num_object_points, controller_points
    ):
        with torch.no_grad():
            # Calculate the force of the springs
            x1 = controller_points[springs[:, 0]]
            x2 = x[springs[:, 1]]

            dis = x2 - x1
            dis_len = torch.norm(dis, dim=1)

            d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
            spring_forces = (
                torch.clamp(spring_Y, min=cfg.spring_Y_min, max=cfg.spring_Y_max)[
                    :, None
                ]
                * (dis_len / rest_lengths - 1.0)[:, None]
                * d
            )

            total_force = -spring_forces.sum(dim=0)
        return total_force

    def visualize_material(self, model_path, gs_path, relative_material=True):
        # Load the model
        logger.info(f"Load model from {model_path}")
        checkpoint = torch.load(model_path, map_location=cfg.device)

        spring_Y = checkpoint["spring_Y"]
        collide_elas = checkpoint["collide_elas"]
        collide_fric = checkpoint["collide_fric"]
        collide_object_elas = checkpoint["collide_object_elas"]
        collide_object_fric = checkpoint["collide_object_fric"]
        num_object_springs = checkpoint["num_object_springs"]

        assert (
            len(spring_Y) == self.simulator.n_springs
        ), "Check if the loaded checkpoint match the config file to connect the springs"

        self.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())
        self.simulator.set_collide(
            collide_elas.detach().clone(), collide_fric.detach().clone()
        )
        self.simulator.set_collide_object(
            collide_object_elas.detach().clone(),
            collide_object_fric.detach().clone(),
        )

        video_path = f"{cfg.base_dir}/material_visualization.mp4"

        vis_cam_idx = 0
        FPS = cfg.FPS
        width, height = cfg.WH
        intrinsic = cfg.intrinsics[vis_cam_idx]
        w2c = cfg.w2cs[vis_cam_idx]

        gaussians = GaussianModel(sh_degree=3)
        gaussians.load_ply(gs_path)
        gaussians = remove_gaussians_with_low_opacity(gaussians, 0.1)
        gaussians.isotropic = True
        current_pos = gaussians.get_xyz
        current_rot = gaussians.get_rotation
        use_white_background = True  # set to True for white background
        bg_color = [1, 1, 1] if use_white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=cfg.device)
        view = self._create_gs_view(w2c, intrinsic, height, width)
        prev_x = None
        relations = None
        weights = None

        # Start to visualize the stuffs
        logger.info("Visualizing the simulation")
        # Visualize the whole simulation using current set of parameters in the physical simulator
        frame_len = self.dataset.frame_len
        self.simulator.set_init_state(
            self.simulator.wp_init_vertices, self.simulator.wp_init_velocities
        )
        prev_x = wp.to_torch(
            self.simulator.wp_states[0].wp_x, requires_grad=False
        ).clone()

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/0.png"
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = render_gaussian(view, gaussians, None, background)
        rendering = results["render"]  # (4, H, W)
        image = rendering.permute(1, 2, 0).detach().cpu().numpy()

        image = image.clip(0, 1)
        if use_white_background:
            image_mask = np.logical_and(
                (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        else:
            image_mask = np.logical_and(
                (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
            )
        image[~image_mask, 3] = 0

        alpha = image[..., 3:4]
        rgb = image[..., :3] * 255
        frame = alpha * rgb + (1 - alpha) * frame
        frame = frame.astype(np.uint8)

        # Add the material visualization
        object_springs = self.init_springs[:num_object_springs]
        material_field = torch.zeros((self.num_all_points, 3), device=cfg.device)
        count_field = torch.zeros(
            self.num_all_points, dtype=torch.int32, device=cfg.device
        )
        clamp_object_spring_Y = torch.clamp(
            spring_Y[:num_object_springs], min=cfg.spring_Y_min, max=cfg.spring_Y_max
        )
        object_rest_lengths = self.init_rest_lengths[:num_object_springs]

        # idx1 = object_springs[:, 0]
        # idx2 = object_springs[:, 1]
        # x1 = prev_x[idx1]
        # x2 = prev_x[idx2]
        # dis = x2 - x1
        # dis_len = torch.norm(dis, dim=1)
        # d = dis / torch.clamp(dis_len, min=1e-6)[:, None]
        # # import pdb
        # # pdb.set_trace()
        # material_field.index_add_(
        #     0,
        #     idx1,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field.index_add_(
        #     0,
        #     idx2,
        #     clamp_object_spring_Y[:, None] / object_rest_lengths[:, None] * d,
        # )
        # material_field = torch.norm(material_field, dim=1)
        # import pdb
        # pdb.set_trace()
        # count_field.index_add_(
        #     0, idx1, torch.ones_like(idx1, dtype=torch.int32, device=cfg.device)
        # )
        # count_field.index_add_(
        #     0, idx2, torch.ones_like(idx2, dtype=torch.int32, device=cfg.device)
        # )
        # material_field /= count_field
        # if relative_material:
        #     material_field_normalized = (material_field - material_field.min()) / (
        #         material_field.max() - material_field.min()
        #     )
        # else:
        #     material_field_normalized = (material_field - cfg.spring_Y_min) / (
        #         cfg.spring_Y_max - cfg.spring_Y_min
        #     )
        # rainbow_colors = plt.cm.rainbow(material_field_normalized.cpu().numpy())[:, :3]

        stiffness_map = compute_effective_stiffness(
            points=prev_x,
            springs=object_springs,
            Y=clamp_object_spring_Y,
            rest_lengths=object_rest_lengths,
            device=cfg.device,
        )
        normed = (stiffness_map - stiffness_map.min()) / (
            stiffness_map.max() - stiffness_map.min()
        )
        rainbow_colors = plt.cm.rainbow(normed.cpu().numpy())[:, :3]

        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
        object_pcd.colors = o3d.utility.Vector3dVector(rainbow_colors)
        vis.add_geometry(object_pcd)

        # Adjust the viewpoint
        view_control = vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            width, height, intrinsic
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

        material_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        material_image = (material_image * 255).astype(np.uint8)
        material_vis_mask = np.all(material_image == [255, 255, 255], axis=-1)
        frame[~material_vis_mask] = material_image[~material_vis_mask]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Interactive Playground", frame)
        cv2.waitKey(1)
        video_writer.write(frame)

        for i in tqdm(range(1, frame_len)):
            if cfg.data_type == "real":
                self.simulator.set_controller_target(i, pure_inference=True)
            if self.simulator.object_collision_flag:
                self.simulator.update_collision_graph()

            wp.capture_launch(self.simulator.forward_graph)
            x = wp.to_torch(self.simulator.wp_states[-1].wp_x, requires_grad=False)
            # Set the intial state for the next step
            self.simulator.set_init_state(
                self.simulator.wp_states[-1].wp_x,
                self.simulator.wp_states[-1].wp_v,
            )

            torch.cuda.synchronize()

            with torch.no_grad():
                # Do LBS on the gaussian kernels
                prev_particle_pos = prev_x
                cur_particle_pos = x
                if relations is None:
                    relations = get_topk_indices(
                        prev_x, K=16
                    )  # only computed in the first iteration

                if weights is None:
                    weights, weights_indices = knn_weights_sparse(
                        prev_particle_pos, current_pos, K=16
                    )  # only computed in the first iteration

                weights = calc_weights_vals_from_indices(
                    prev_particle_pos, current_pos, weights_indices
                )

                current_pos, current_rot, _ = interpolate_motions_speedup(
                    bones=prev_particle_pos,
                    motions=cur_particle_pos - prev_particle_pos,
                    relations=relations,
                    weights=weights,
                    weights_indices=weights_indices,
                    xyz=current_pos,
                    quat=current_rot,
                )

                # update gaussians with the new positions and rotations
                gaussians._xyz = current_pos
                gaussians._rotation = current_rot

            prev_x = x.clone()

            frame_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = render_gaussian(view, gaussians, None, background)
            rendering = results["render"]  # (4, H, W)
            image = rendering.permute(1, 2, 0).detach().cpu().numpy()

            image = image.clip(0, 1)
            if use_white_background:
                image_mask = np.logical_and(
                    (image != 1.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            else:
                image_mask = np.logical_and(
                    (image != 0.0).any(axis=2), image[:, :, 3] > 100 / 255
                )
            image[~image_mask, 3] = 0

            alpha = image[..., 3:4]
            rgb = image[..., :3] * 255
            frame = alpha * rgb + (1 - alpha) * frame
            frame = frame.astype(np.uint8)

            # Update the object pcd
            object_pcd.points = o3d.utility.Vector3dVector(prev_x.cpu().numpy())
            vis.update_geometry(object_pcd)

            vis.poll_events()
            vis.update_renderer()

            force_image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            force_image = (force_image * 255).astype(np.uint8)
            force_vis_mask = np.all(force_image == [255, 255, 255], axis=-1)
            frame[~force_vis_mask] = force_image[~force_vis_mask]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

            cv2.imshow("Interactive Playground", frame)
            cv2.waitKey(1)
        vis.destroy_window()
        video_writer.release()


def get_simple_shadow(
    points,
    intrinsic,
    w2c,
    width,
    height,
    image_mask,
    kernel_size=7,
    light_point=[0, 0, -3],
):
    points = points.cpu().numpy()

    t = -points[:, 2] / light_point[2]
    points_on_table = points + t[:, None] * light_point

    points_homogeneous = np.hstack(
        [points_on_table, np.ones((points_on_table.shape[0], 1))]
    )  # Convert to homogeneous coordinates
    points_camera = (w2c @ points_homogeneous.T).T

    points_pixels = (intrinsic @ points_camera[:, :3].T).T
    points_pixels /= points_pixels[:, 2:3]
    pixel_coords = points_pixels[:, :2]

    valid_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < width)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < height)
    )

    valid_pixel_coords = pixel_coords[valid_mask]
    valid_pixel_coords = valid_pixel_coords.astype(int)

    shadow_image = np.zeros((height, width), dtype=np.uint8)
    shadow_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel_1 = np.ones((3, 3), np.uint(8))
    dilated_shadow = cv2.dilate(shadow_image, kernel, iterations=1)
    dilated_shadow = cv2.dilate(dilated_shadow, kernel_1, iterations=1)
    final_shadow = cv2.erode(dilated_shadow, kernel, iterations=1)

    final_shadow[image_mask] = 0
    final_shadow = final_shadow == 255
    return final_shadow


# Borrow ideas and codes from H. Sánchez's answer
# https://stackoverflow.com/questions/59026581/create-arrows-in-open3d
def getArrowMesh(origin=[0, 0, 0], end=None, color=[0, 0, 0]):
    vec_Arr = np.array(end) - np.array(origin)
    vec_len = np.linalg.norm(vec_Arr)
    mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cone_height=0.05 * vec_len,
        cone_radius=0.002,
        cylinder_height=0.2 * vec_len,
        cylinder_radius=0.003,
    )
    mesh_arrow.paint_uniform_color(color)
    rot_mat = _caculate_align_mat(vec_Arr / vec_len)
    mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
    mesh_arrow.translate(np.array(origin))
    return mesh_arrow


def _get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array(
        [
            [0, -pVec_Arr[2], pVec_Arr[1]],
            [pVec_Arr[2], 0, -pVec_Arr[0]],
            [-pVec_Arr[1], pVec_Arr[0], 0],
        ]
    )
    return qCross_prod_mat


def _caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = _get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = _get_cross_prod_mat(z_c_vec)
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = (
            np.eye(3, 3)
            + z_c_vec_mat
            + np.matmul(z_c_vec_mat, z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
        )
    qTrans_Mat *= scale
    return qTrans_Mat


def construct_stiffness_matrix_sparse(
    springs, positions, spring_Y, rest_lengths, num_points, device
):
    # springs: (N_springs, 2)
    # positions: (N_points, 3)
    # spring_Y: (N_springs,)
    # rest_lengths: (N_springs,)

    i = springs[:, 0]
    j = springs[:, 1]

    x_i = positions[i]  # (N, 3)
    x_j = positions[j]
    d = x_j - x_i  # (N, 3)
    d_norm = torch.norm(d, dim=1, keepdim=True) + 1e-8
    d_hat = d / d_norm  # (N, 3)

    coeff = spring_Y / rest_lengths  # (N,)
    k_blocks = coeff[:, None, None] * (
        d_hat[:, :, None] @ d_hat[:, None, :]
    )  # (N, 3, 3)

    indices = []
    values = []

    for shift_i, shift_j, sign in [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]:
        node_i = springs[:, shift_i]
        node_j = springs[:, shift_j]

        for a in range(3):
            for b in range(3):
                row_idx = 3 * node_i + a
                col_idx = 3 * node_j + b
                val = sign * k_blocks[:, a, b]
                indices.append(torch.stack([row_idx, col_idx], dim=0))  # (2, N)
                values.append(val)

    indices = torch.cat(indices, dim=1)  # (2, total_nonzero)
    values = torch.cat(values, dim=0)  # (total_nonzero,)
    size = (3 * num_points, 3 * num_points)
    K_sparse = torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()
    return K_sparse


def compute_effective_stiffness(points, springs, Y, rest_lengths, device):
    """
    Compute effective stiffness for each point based on stiffness matrix diagonal blocks.
    Return: (N_points,) tensor of Frobenius norm of 3x3 diagonal blocks in stiffness matrix.
    """
    num_points = points.shape[0]
    K_sparse = construct_stiffness_matrix_sparse(
        springs=springs,
        positions=points,
        spring_Y=Y,
        rest_lengths=rest_lengths,
        num_points=num_points,
        device=device,
    )

    K_dense = K_sparse.to_dense()
    stiffness_map = torch.zeros(num_points, device=device)
    for i in range(num_points):
        block = K_dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]
        stiffness_map[i] = torch.norm(block, p="fro")
    return stiffness_map
```

qqtt/env/__init__.py
```python
from .camera import CameraSystem```

qqtt/env/camera/__init__.py
```python
from .camera_system import CameraSystem```

qqtt/env/camera/camera_system.py
```python
from .realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
from pynput import keyboard
import cv2
import json
import os
import pickle

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class CameraSystem:
    def __init__(
        self, WH=[848, 480], fps=30, num_cam=3, exposure=50, gain=60, white_balance=3800
    ):
        self.WH = WH
        self.fps = fps

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        self.num_cam = len(self.serial_numbers)
        assert self.num_cam == num_cam, f"Only {self.num_cam} cameras are connected."

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.fps,
            enable_color=True,
            enable_depth=True,
            process_depth=True,
            verbose=False,
        )
        # Some camera settings
        self.realsense.set_exposure(exposure=exposure, gain=gain)
        self.realsense.set_white_balance(white_balance)

        self.realsense.start()
        time.sleep(3)
        self.recording = False
        self.end = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("Camera system is ready.")

    def get_observation(self):
        # Used to get the latest observations from all cameras
        data = self._get_sync_frame()
        # TODO: Process the data when needed
        return data

    def _get_sync_frame(self, k=4):
        assert self.realsense.is_ready

        # Get the latest k frames from all cameras, and picked the latest synchronized frames
        last_realsense_data = self.realsense.get(k=k)
        timestamp_list = [x["timestamp"][-1] for x in last_realsense_data.values()]
        last_timestamp = np.min(timestamp_list)

        data = {}
        for camera_idx, value in last_realsense_data.items():
            this_timestamps = value["timestamp"]
            min_diff = 10
            best_idx = None
            for i, this_timestamp in enumerate(this_timestamps):
                diff = np.abs(this_timestamp - last_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    best_idx = i
            # remap key, step_idx is different, timestamp can be the same when some frames are lost
            data[camera_idx] = {}
            data[camera_idx]["color"] = value["color"][best_idx]
            data[camera_idx]["depth"] = value["depth"][best_idx]
            data[camera_idx]["timestamp"] = value["timestamp"][best_idx]
            data[camera_idx]["step_idx"] = value["step_idx"][best_idx]

        return data

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if self.recording == False:
                    self.recording = True
                    print("Start recording")
                else:
                    self.recording = False
                    self.end = True
        except AttributeError:
            pass

    def record(self, output_path):
        exist_dir(output_path)
        exist_dir(f"{output_path}/color")
        exist_dir(f"{output_path}/depth")

        metadata = {}
        intrinsics = self.realsense.get_intrinsics()
        metadata["intrinsics"] = intrinsics.tolist()
        metadata["serial_numbers"] = self.serial_numbers
        metadata["fps"] = self.fps
        metadata["WH"] = self.WH
        metadata["recording"] = {}
        for i in range(self.num_cam):
            metadata["recording"][i] = {}
            exist_dir(f"{output_path}/color/{i}")
            exist_dir(f"{output_path}/depth/{i}")

        # Set the max time for recording
        last_step_idxs = [-1] * self.num_cam
        while not self.end:
            if self.recording:
                last_realsense_data = self.realsense.get()
                timestamps = [
                    last_realsense_data[i]["timestamp"].item()
                    for i in range(self.num_cam)
                ]
                step_idxs = [
                    last_realsense_data[i]["step_idx"].item()
                    for i in range(self.num_cam)
                ]

                if not all(
                    [step_idxs[i] == last_step_idxs[i] for i in range(self.num_cam)]
                ):
                    for i in range(self.num_cam):
                        if last_step_idxs[i] != step_idxs[i]:
                            # Record the the step for this camera
                            time_stamp = timestamps[i]
                            step_idx = step_idxs[i]
                            color = last_realsense_data[i]["color"]
                            depth = last_realsense_data[i]["depth"]

                            metadata["recording"][i][step_idx] = time_stamp
                            cv2.imwrite(
                                f"{output_path}/color/{i}/{step_idx}.png", color
                            )
                            np.save(f"{output_path}/depth/{i}/{step_idx}.npy", depth)

        print("End recording")
        self.listener.stop()
        with open(f"{output_path}/metadata.json", "w") as f:
            json.dump(metadata, f)

        self.realsense.stop()

    def calibrate(self, visualize=True):
        # Initialize the calibration board information
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard(
            (4, 5),
            squareLength=0.05,
            markerLength=0.037,
            dictionary=dictionary,
        )
        # Get the intrinsic information from the realsense camera
        intrinsics = self.realsense.get_intrinsics()

        flag = True
        while flag:
            flag = False
            obs = self.get_observation()
            colors = [obs[i]["color"] for i in range(self.num_cam)]

            c2ws = []
            for i in range(self.num_cam):
                intrinsic = intrinsics[i]
                calibration_img = colors[i]
                # cv2.imshow("cablibration", calibration_img)
                # cv2.waitKey(0)

                corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                    image=calibration_img,
                    dictionary=dictionary,
                    parameters=None,
                )
                retval, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=calibration_img,
                        board=board,
                        cameraMatrix=intrinsic,
                    )
                )
                # cv2.imshow("cablibration", calibration_img)

                print("number of corners: ", len(charuco_corners))
                if visualize:
                    cv2.aruco.drawDetectedCornersCharuco(
                        image=calibration_img,
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                    )
                    cv2.imshow("cablibration", calibration_img)
                    cv2.waitKey(1)

                rvec = None
                tvec = None
                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board,
                    intrinsic,
                    None,
                    rvec=rvec,
                    tvec=tvec,
                )

                # Reproject the points to calculate the error
                reprojected_points, _ = cv2.projectPoints(
                    board.getChessboardCorners()[charuco_ids, :],
                    rvec,
                    tvec,
                    intrinsic,
                    None,
                )
                # Reshape for easier handling
                reprojected_points = reprojected_points.reshape(-1, 2)
                charuco_corners = charuco_corners.reshape(-1, 2)
                # Calculate the error
                error = np.sqrt(
                    np.sum((reprojected_points - charuco_corners) ** 2, axis=1)
                ).mean()

                print("Reprojection Error:", error)
                if error > 0.2 or len(charuco_corners) < 11:
                    flag = True
                    print("Please try again.")
                    break
                R_board2cam = cv2.Rodrigues(rvec)[0]
                t_board2cam = tvec[:, 0]
                w2c = np.eye(4)
                w2c[:3, :3] = R_board2cam
                w2c[:3, 3] = t_board2cam
                c2ws.append(np.linalg.inv(w2c))

        with open("calibrate.pkl", "wb") as f:
            pickle.dump(c2ws, f)

        self.realsense.stop()
```

qqtt/env/camera/realsense/__init__.py
```python
from .multi_realsense import MultiRealsense, SingleRealsense```

qqtt/env/camera/realsense/multi_realsense.py
```python
# Description: MultiRealsense class for multiple RealSense cameras, based on code from Diffusion Policy

from typing import List, Optional, Union, Dict
from collections.abc import Callable
import numbers
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from .single_realsense import SingleRealsense

class MultiRealsense:
    def __init__(self,
        serial_numbers: list[str] | None=None,
        shm_manager: SharedMemoryManager | None=None,
        resolution=(1280,720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        enable_color=True,
        enable_depth=False,
        process_depth=False,
        enable_infrared=False,
        get_max_k=30,
        advanced_mode_config: dict | list[dict] | None=None,
        transform: Callable[[dict], dict] | list[Callable] | None=None,
        vis_transform: Callable[[dict], dict] | list[Callable] | None=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)

        cameras = dict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                is_master=(i == 0),
                verbose=verbose
            )
        
        self.cameras = cameras
        self.serial_numbers = serial_numbers
        self.shm_manager = shm_manager
        self.resolution = resolution
        self.capture_fps = capture_fps

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            print(f'processing camera {camera.serial_number}')
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, index=None, out=None) -> dict[int, dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if index is not None:
            this_out = None
            this_out = self.cameras[self.serial_numbers[index]].get(k=k, out=this_out)
            return this_out
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """150nit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        return np.array([c.get_intrinsics() for c in self.cameras.values()])

    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x
```

qqtt/env/camera/realsense/shared_memory/__init__.py
```python
```

qqtt/env/camera/realsense/shared_memory/shared_memory_queue.py
```python
from typing import Dict, List, Union
import numbers
from queue import Empty, Full
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from .shared_memory_util import ArraySpec, SharedAtomicCounter
from .shared_ndarray import SharedNDArray


class SharedMemoryQueue:
    """
    A Lock-Free FIFO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            array_specs: list[ArraySpec],
            buffer_size: int
        ):

        # create atomic counter
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)
        
        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays
    
    @classmethod
    def create_from_examples(cls, 
            shm_manager: SharedMemoryManager,
            examples: dict[str, np.ndarray | numbers.Number], 
            buffer_size: int
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            buffer_size=buffer_size
            )
        return obj
    
    def qsize(self):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        return n_data
    
    def empty(self):
        n_data = self.qsize()
        return n_data <= 0
    
    def clear(self):
        self.read_counter.store(self.write_counter.load())
    
    def put(self, data: dict[str, np.ndarray | numbers.Number]):
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)
    
    def get(self, out=None) -> dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])
        
        # update idx
        self.read_counter.add(1)
        return out

    def get_k(self, k, out=None) -> dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> dict[str, np.ndarray]:
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out
    
    def _get_k_impl(self, k, read_count, out=None) -> dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start: target_end] = arr[start:end]

        return out
    
    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result
```

qqtt/env/camera/realsense/shared_memory/shared_memory_ring_buffer.py
```python
from typing import Dict, List, Union

from queue import Empty
import numbers
import time
from multiprocessing.managers import SharedMemoryManager
import numpy as np

from .shared_ndarray import SharedNDArray
from .shared_memory_util import ArraySpec, SharedAtomicCounter

class SharedMemoryRingBuffer:
    """
    A Lock-Free FILO Shared Memory Data Structure.
    Stores a sequence of dict of numpy arrays.
    """

    def __init__(self, 
            shm_manager: SharedMemoryManager,
            array_specs: list[ArraySpec],
            get_max_k: int,
            get_time_budget: float,
            put_desired_frequency: float,
            safety_margin: float=10
        ):
        """
        shm_manager: Manages the life cycle of share memories 
            across processes. Remember to run .start() before passing.
        array_specs: Name, shape and type of arrays for a single time step.
        get_max_k: The maxmum number of items can be queried at once.
        get_time_budget: The maxmum amount of time spent copying data from 
            shared memory to local memory. Increase this number for larger arrays.
        put_desired_frequency: The maximum frequency that .put() can be called.
            This influces the buffer size.
        """

        # create atomic counter
        counter = SharedAtomicCounter(shm_manager)

        # compute buffer size
        # At any given moment, the past get_max_k items should never 
        # be touched (to be read freely). Assuming the reading is reading
        # these k items, which takes maximum of get_time_budget seconds,
        # we need enough empty slots to make sure put_desired_frequency Hz
        # of put can be sustaied.
        buffer_size = int(np.ceil(
            put_desired_frequency * get_time_budget 
            * safety_margin)) + get_max_k

        # allocate shared memory
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        # allocate timestamp array
        timestamp_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager, 
            shape=(buffer_size,),
            dtype=np.float64)
        timestamp_array.get()[:] = -np.inf
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.counter = counter
        self.shared_arrays = shared_arrays
        self.timestamp_array = timestamp_array
        self.get_time_budget = get_time_budget
        self.get_max_k = get_max_k
        self.put_desired_frequency = put_desired_frequency
        self.ready_for_get = False

    
    @property
    def count(self):
        return self.counter.load()
    
    @classmethod
    def create_from_examples(cls, 
            shm_manager: SharedMemoryManager,
            examples: dict[str, np.ndarray | numbers.Number], 
            get_max_k: int=32,
            get_time_budget: float=0.01,
            put_desired_frequency: float=60
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(
            shm_manager=shm_manager,
            array_specs=specs,
            get_max_k=get_max_k,
            get_time_budget=get_time_budget,
            put_desired_frequency=put_desired_frequency
            )
        return obj

    def clear(self):
        self.counter.store(0)
    
    def put(self, data: dict[str, np.ndarray | numbers.Number], wait: bool=True, serial_number: str='unknown'):
        count = self.counter.load()
        next_idx = count % self.buffer_size
        # Make sure the next self.get_max_k elements in the ring buffer have at least 
        # self.get_time_budget seconds untouched after written, so that
        # get_last_k can safely read k elements from any count location.
        # Sanity check: when get_max_k == 1, the element pointed by next_idx
        # should be rewritten at minimum self.get_time_budget seconds later.
        timestamp_lookahead_idx = (next_idx + self.get_max_k - 1) % self.buffer_size
        old_timestamp = self.timestamp_array.get()[timestamp_lookahead_idx]
        t = time.monotonic()
        if (t - old_timestamp) < self.get_time_budget:
            deltat = t - old_timestamp
            if wait:
                # sleep the remaining time to be safe
                time.sleep(self.get_time_budget - deltat)
            else:
                if self.ready_for_get:
                    # throw an error
                    past_iters = self.buffer_size - self.get_max_k
                    hz = past_iters / deltat
                    raise TimeoutError(
                        '[Camera {}] Put executed too fast {}items/{:.4f}s ~= {}Hz'.format(
                            serial_number, past_iters, deltat,hz))

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)
        
        # update timestamp
        self.timestamp_array.get()[next_idx] = time.monotonic()
        self.counter.add(1)

    def _allocate_empty(self, k=None):
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result

    def get(self, out=None) -> dict[str, np.ndarray]:
        if out is None:
            out = self._allocate_empty()
        start_time = time.monotonic()
        count = self.counter.load()
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[curr_idx])
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out
    
    def get_last_k(self, k:int, out=None) -> dict[str, np.ndarray]:
        assert k <= self.get_max_k
        if out is None:
            out = self._allocate_empty(k)
        start_time = time.monotonic()
        count = self.counter.load()
        assert k <= count
        curr_idx = (count - 1) % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            end = curr_idx + 1
            start = max(0, end - k)
            target_end = k
            target_start = target_end - (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                end = self.buffer_size
                start = end - remainder
                target_start = 0
                target_end = end - start
                target[target_start: target_end] = arr[start:end]
        end_time = time.monotonic()
        dt = end_time - start_time
        if dt > self.get_time_budget:
            raise TimeoutError(f'Get time out {dt} vs {self.get_time_budget}')
        return out

    def get_all(self) -> dict[str, np.ndarray]:
        k = min(self.count, self.get_max_k)
        return self.get_last_k(k=k)
```

qqtt/env/camera/realsense/shared_memory/shared_memory_util.py
```python
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from atomics import atomicview, MemoryOrder, UINT

@dataclass
class ArraySpec:
    name: str
    shape: tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, 
            shm_manager: SharedMemoryManager, 
            size :int=8 # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0) # initialize

    @property
    def buf(self):
        return self.shm.buf[:self.size]

    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a: 
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value
    
    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)
    
    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
```

qqtt/env/camera/realsense/shared_memory/shared_ndarray.py
```python
from __future__ import annotations

import multiprocessing
import multiprocessing.synchronize
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Any, TYPE_CHECKING, Generic, Optional, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt


SharedMemoryLike = Union[str, SharedMemory]  # shared memory or name of shared memory
SharedT = TypeVar("SharedT", bound=np.generic)


class SharedNDArray(Generic[SharedT]):
    """Class to keep track of and retrieve the data in a shared array
    Attributes
    ----------
    shm
        SharedMemory object containing the data of the array
    shape
        Shape of the NumPy array
    dtype
        Type of the NumPy array. Anything that may be passed to the `dtype=` argument in `np.ndarray`.
    lock
        (Optional) multiprocessing.Lock to manage access to the SharedNDArray. This is only created if
        lock=True is passed to the constructor, otherwise it is set to `None`.
    A SharedNDArray object may be created either directly with a preallocated shared memory object plus the
    dtype and shape of the numpy array it represents:
    >>> from multiprocessing.shared_memory import SharedMemory
    >>> import numpy as np
    >>> from shared_ndarray2 import SharedNDArray
    >>> x = np.array([1, 2, 3])
    >>> shm = SharedMemory(name="x", create=True, size=x.nbytes)
    >>> arr = SharedNDArray(shm, x.shape, x.dtype)
    >>> arr[:] = x[:]  # copy x into the array
    >>> print(arr[:])
    [1 2 3]
    >>> shm.close()
    >>> shm.unlink()
    Or using a SharedMemoryManager either from an existing array or from arbitrary shape and nbytes:
    >>> from multiprocessing.managers import SharedMemoryManager
    >>> mem_mgr = SharedMemoryManager()
    >>> mem_mgr.start()  # Better yet, use SharedMemoryManager context manager
    >>> arr = SharedNDArray.from_shape(mem_mgr, x.shape, x.dtype)
    >>> arr[:] = x[:]  # copy x into the array
    >>> print(arr[:])
    [1 2 3]
    >>> # -or in one step-
    >>> arr = SharedNDArray.from_array(mem_mgr, x)
    >>> print(arr[:])
    [1 2 3]
    `SharedNDArray` does not subclass numpy.ndarray but rather generates an ndarray on-the-fly in get(),
    which is used in __getitem__ and __setitem__. Thus to access the data and/or use any ndarray methods
    get() or __getitem__ or __setitem__ must be used
    >>> arr.max()  # ERROR: SharedNDArray has no `max` method.
    Traceback (most recent call last):
        ....
    AttributeError: SharedNDArray object has no attribute 'max'. To access NumPy ndarray object use .get() method.
    >>> arr.get().max()  # (or arr[:].max())  OK: This gets an ndarray on which we can operate
    3
    >>> y = np.zeros(3)
    >>> y[:] = arr  # ERROR: Cannot broadcast-assign a SharedNDArray to ndarray `y`
    Traceback (most recent call last):
        ...
    ValueError: setting an array element with a sequence.
    >>> y[:] = arr[:]  # OK: This gets an ndarray that can be copied element-wise to `y`
    >>> mem_mgr.shutdown()
    """

    shm: SharedMemory
    # shape: Tuple[int, ...]  # is a property
    dtype: np.dtype
    lock: multiprocessing.synchronize.Lock | None

    def __init__(
        self, shm: SharedMemoryLike, shape: tuple[int, ...], dtype: npt.DTypeLike):
        """Initialize a SharedNDArray object from existing shared memory, object shape, and dtype.
        To initialize a SharedNDArray object from a memory manager and data or shape, use the `from_array()
        or `from_shape()` classmethods.
        Parameters
        ----------
        shm
            `multiprocessing.shared_memory.SharedMemory` object or name for connecting to an existing block
            of shared memory (using SharedMemory constructor)
        shape
            Shape of the NumPy array to be represented in the shared memory
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        lock : bool, optional
            If True, create a multiprocessing.Lock object accessible with the `.lock` attribute, by default
            False.  If passing the `SharedNDArray` as an argument to a `multiprocessing.Pool` function this
            should not be used -- see this comment to a Stack Overflow question about `multiprocessing.Lock`:
            https://stackoverflow.com/questions/25557686/python-sharing-a-lock-between-processes#comment72803059_25558333
        Raises
        ------
        ValueError
            The SharedMemory size (number of bytes) does not match the product of the shape and dtype
            itemsize.
        """
        if isinstance(shm, str):
            shm = SharedMemory(name=shm, create=False)
        dtype = np.dtype(dtype)  # Try to convert to dtype
        assert shm.size >= (dtype.itemsize * np.prod(shape))
        self.shm = shm
        self.dtype = dtype
        self._shape: tuple[int, ...] = shape

    def __repr__(self):
        # Like numpy's ndarray repr
        cls_name = self.__class__.__name__
        nspaces = len(cls_name) + 1
        array_repr = str(self.get())
        array_repr = array_repr.replace("\n", "\n" + " " * nspaces)
        return f"{cls_name}({array_repr}, dtype={self.dtype})"

    @classmethod
    def create_from_array(
        cls, mem_mgr: SharedMemoryManager, arr: npt.NDArray[SharedT]
    ) -> SharedNDArray[SharedT]:
        """Create a SharedNDArray from a SharedMemoryManager and an existing numpy array.
        Parameters
        ----------
        mem_mgr
            Running `multiprocessing.managers.SharedMemoryManager` instance from which to create the
            SharedMemory for the SharedNDArray
        arr
            NumPy `ndarray` object to copy into the created SharedNDArray upon initialization.
        """
        # Simply use from_shape() to create the SharedNDArray and copy the data into it.
        shared_arr = cls.create_from_shape(mem_mgr, arr.shape, arr.dtype)
        shared_arr.get()[:] = arr[:]
        return shared_arr

    @classmethod
    def create_from_shape(
        cls, mem_mgr: SharedMemoryManager, shape: tuple, dtype: npt.DTypeLike) -> SharedNDArray:
        """Create a SharedNDArray directly from a SharedMemoryManager
        Parameters
        ----------
        mem_mgr
            SharedMemoryManager instance that has been started
        shape
            Shape of the array
        dtype
            Data type for the NumPy array to be represented in shared memory. Any valid argument for
            `np.dtype` may be used as it will be converted to an actual `dtype` object.
        """
        dtype = np.dtype(dtype)  # Convert to dtype if possible
        shm = mem_mgr.SharedMemory(np.prod(shape) * dtype.itemsize)
        return cls(shm=shm, shape=shape, dtype=dtype)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape


    def get(self) -> npt.NDArray[SharedT]:
        """Get a numpy array with access to the shared memory"""
        return np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)

    def __del__(self):
        self.shm.close()
```

qqtt/env/camera/realsense/single_realsense.py
```python
# Description: MultiRealsense class for multiple RealSense cameras, based on code from Diffusion Policy

from typing import Optional, Dict
from collections.abc import Callable
import os
import enum
import time
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

from .utils import get_accumulate_timestamp_idxs
from .shared_memory.shared_ndarray import SharedNDArray
from .shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from .shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty

class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            enable_color=True,
            enable_depth=False,
            process_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Callable[[dict], dict] | None = None,
            vis_transform: Callable[[dict], dict] | None = None,
            is_master=False,
            verbose=False
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=shape, dtype=np.uint16)
        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.process_depth = process_depth
        self.is_master = is_master
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
    
    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
    
    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def depth_process(self, depth_frame):
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
        
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.75)
        spatial.set_option(rs.option.filter_smooth_delta, 1)
        spatial.set_option(rs.option.holes_fill, 1)
        
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.75)
        temporal.set_option(rs.option.filter_smooth_delta, 1)

        filtered_depth = depth_to_disparity.process(depth_frame)
        filtered_depth = spatial.process(filtered_depth)
        filtered_depth = temporal.process(filtered_depth)
        filtered_depth = disparity_to_depth.process(filtered_depth)
        return filtered_depth

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })
     
    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)
        w, h = self.resolution
        fps = self.capture_fps
        align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, 
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth, 
                w, h, rs.format.z16, fps)
        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)
        
        def init_device():
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            self.pipeline = pipeline
            self.pipeline_profile = pipeline_profile

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = self.pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, 1)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = self.pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            color_stream = self.pipeline_profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = getattr(intr, name)

            if self.enable_depth:
                depth_sensor = self.pipeline_profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
                self.intrinsics_array.get()[-1] = depth_scale
            
            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

        try:
            init_device()
            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            while not self.stop_event.is_set():
                # wait for frames to come in
                frameset = None
                while frameset is None:
                    try:
                        frameset = self.pipeline.wait_for_frames()
                    except RuntimeError as e:
                        print(f'[SingleRealsense {self.serial_number}] Error: {e}. Ready state: {self.ready_event.is_set()}, Restarting device.')
                        device = self.pipeline.get_active_profile().get_device()
                        device.hardware_reset()
                        self.pipeline.stop()
                        init_device()
                        continue
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)

                self.ring_buffer.ready_for_get = (receive_time - put_start_time >= 0)

                # grab data
                if self.verbose:
                    grad_start_time = time.time()
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    # print(time.time())
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    depth_frame = frameset.get_depth_frame()
                    if self.process_depth:
                        data['depth'] = self.depth_process(depth_frame).get_data()
                    else:
                        data['depth'] = np.asarray(depth_frame.get_data())
                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Grab data time {time.time() - grad_start_time}')
                
                # apply transform
                if self.verbose:
                    transform_start_time = time.time()
                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Transform time {time.time() - transform_start_time}')

                if self.verbose:
                    put_data_start_time = time.time()
                if self.put_downsample:                
                    # put frequency regulation
                    # print(self.serial_number, put_start_time, put_idx, len(global_idxs))
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False, serial_number=self.serial_number)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    print(step_idx, receive_time)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False, serial_number=self.serial_number)
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Put data time {time.time() - put_data_start_time}', end=' ')
                    print(f'with downsample for {len(global_idxs)}x' if self.put_downsample and len(global_idxs) > 1 else '')

                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = self.pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = self.pipeline_profile.get_device().first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1 if self.is_master else 2)
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']

                iter_idx += 1
        finally:
            rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')
```

qqtt/env/camera/realsense/utils.py
```python
# Description: MultiRealsense class for multiple RealSense cameras, based on code from Diffusion Policy

from typing import List, Tuple, Optional, Dict
import math
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: list[float],  
    start_time: float, 
    dt: float, 
    eps:float=1e-5,
    next_global_idx: int | None=0,
    allow_negative=False
    ) -> tuple[list[int], list[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx. 
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """
    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt 
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(    
        timestamps: list[float], 
        target_global_idxs: list[int], 
        start_time: float, 
        dt: float, 
        eps:float=1e-5):
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True
    )
    if len(global_idxs) > len(target_global_idxs):
        # if more steps available, truncate
        global_idxs = global_idxs[:len(target_global_idxs)]
        local_idxs = local_idxs[:len(target_global_idxs)]
    
    if len(global_idxs) == 0:
        import pdb; pdb.set_trace()

    for i in range(len(target_global_idxs) - len(global_idxs)):
        # if missing, repeat
        local_idxs.append(len(timestamps)-1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs


class TimestampObsAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.obs_buffer = dict()
        self.timestamp_buffer = None
        self.next_global_idx = 0
    
    def __len__(self):
        return self.next_global_idx
    
    @property
    def data(self):
        if self.timestamp_buffer is None:
            return dict()
        result = dict()
        for key, value in self.obs_buffer.items():
            result[key] = value[:len(self)]
        return result

    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, data: dict[str, np.ndarray], timestamps: np.ndarray):
        """
        data:
            key: T,*
        """

        local_idxs, global_idxs, self.next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.obs_buffer = dict()
                for key, value in data.items():
                    self.obs_buffer[key] = np.zeros_like(value)
                self.timestamp_buffer = np.zeros(
                    (len(timestamps),), dtype=np.float64)
            
            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                for key in list(self.obs_buffer.keys()):
                    new_shape = (new_size,) + self.obs_buffer[key].shape[1:]
                    self.obs_buffer[key] = np.resize(self.obs_buffer[key], new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size))
            
            # write data
            for key, value in self.obs_buffer.items():
                value[global_idxs] = data[key][local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]


class TimestampActionAccumulator:
    def __init__(self, 
            start_time: float, 
            dt: float, 
            eps: float=1e-5):
        """
        Different from Obs accumulator, the action accumulator
        allows overwriting previous values.
        """
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.action_buffer = None
        self.timestamp_buffer = None
        self.size = 0
    
    def __len__(self):
        return self.size
    
    @property
    def actions(self):
        if self.action_buffer is None:
            return np.array([])
        return self.action_buffer[:len(self)]
    
    @property
    def actual_timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[:len(self)]
    
    @property
    def timestamps(self):
        if self.timestamp_buffer is None:
            return np.array([])
        return self.start_time + np.arange(len(self)) * self.dt

    def put(self, actions: np.ndarray, timestamps: np.ndarray):
        """
        Note: timestamps is the time when the action will be issued, 
        not when the action will be completed (target_timestamp)
        """

        local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
            timestamps=timestamps,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            # allows overwriting previous actions
            next_global_idx=None
        )

        if len(global_idxs) > 0:
            if self.timestamp_buffer is None:
                # first allocation
                self.action_buffer = np.zeros_like(actions)
                self.timestamp_buffer = np.zeros((len(actions),), dtype=np.float64)

            this_max_size = global_idxs[-1] + 1
            if this_max_size > len(self.timestamp_buffer):
                # reallocate
                new_size = max(this_max_size, len(self.timestamp_buffer) * 2)
                new_shape = (new_size,) + self.action_buffer.shape[1:]
                self.action_buffer = np.resize(self.action_buffer, new_shape)
                self.timestamp_buffer = np.resize(self.timestamp_buffer, (new_size,))
            
            # potentially rewrite old data (as expected)
            self.action_buffer[global_idxs] = actions[local_idxs]
            self.timestamp_buffer[global_idxs] = timestamps[local_idxs]
            self.size = max(self.size, this_max_size)
```

qqtt/model/__init__.py
```python
from .diff_simulator import SpringMassSystemWarp```

qqtt/model/diff_simulator/__init__.py
```python
from .spring_mass_warp import SpringMassSystemWarp```

qqtt/model/diff_simulator/spring_mass_warp.py
```python
import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_ground = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v = wp.zeros_like(self.wp_x, requires_grad=True)
        self.wp_vertice_forces = wp.zeros_like(self.wp_x, requires_grad=True)
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    control_x: wp.array(dtype=wp.vec3),
    control_v: wp.array(dtype=wp.vec3),
    num_object_points: int,
    springs: wp.array(dtype=wp.vec2i),
    rest_lengths: wp.array(dtype=float),
    spring_Y: wp.array(dtype=float),
    dashpot_damping: float,
    spring_Y_min: float,
    spring_Y_max: float,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    if wp.exp(spring_Y[tid]) > spring_Y_min:

        idx1 = springs[tid][0]
        idx2 = springs[tid][1]

        if idx1 >= num_object_points:
            x1 = control_x[idx1 - num_object_points]
            v1 = control_v[idx1 - num_object_points]
        else:
            x1 = x[idx1]
            v1 = v[idx1]
        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]
            v2 = control_v[idx2 - num_object_points]
        else:
            x2 = x[idx2]
            v2 = v[idx2]

        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )

        v_rel = wp.dot(v2 - v1, d)
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]
    f0 = f[tid]
    m0 = masses[tid]

    drag_damping_factor = wp.exp(-dt * drag_damping)
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    a = all_force / m0
    v1 = v0 + a * dt
    v2 = v1 * drag_damping_factor

    v_new[tid] = v2


@wp.func
def loop(
    i: int,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    clamp_collide_object_elas: float,
    clamp_collide_object_fric: float,
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]

    valid_count = float(0.0)
    J_sum = wp.vec3(0.0, 0.0, 0.0)
    for k in range(collision_number[i]):
        index = collision_indices[i][k]
        x2 = x[index]
        v2 = v[index]
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2
            and dis_len < collision_dist
            and wp.dot(dis, relative_v) < -1e-4
        ):
            valid_count += 1.0

            collision_normal = dis / wp.max(dis_len, 1e-6)
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )
            v_rel_n_length = wp.length(v_rel_n)

            v_rel_t = relative_v - v_rel_n
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            J = impulse_n + impulse_t

            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),
    masks: wp.array(dtype=wp.int32),
    collision_dist: float,
    grid: wp.uint64,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]
    mask1 = masks[i]

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0)
    for index in neighbors:
        if index != i:
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    masks: wp.array(dtype=wp.int32),
    collide_object_elas: wp.array(dtype=float),
    collide_object_fric: wp.array(dtype=float),
    collision_dist: float,
    collision_indices: wp.array2d(dtype=wp.int32),
    collision_number: wp.array(dtype=wp.int32),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v1 = v[tid]
    m1 = masses[tid]

    clamp_collide_object_elas = wp.clamp(collide_object_elas[0], low=0.0, high=1.0)
    clamp_collide_object_fric = wp.clamp(collide_object_fric[0], low=0.0, high=2.0)

    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        J_average = J_sum / valid_count
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    collide_elas: wp.array(dtype=float),
    collide_fric: wp.array(dtype=float),
    dt: float,
    reverse_factor: float,
    x_new: wp.array(dtype=wp.vec3),
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor

    x_z = x0[2]
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # Ground Collision
        v_normal = wp.dot(v0, normal) * normal
        v_tao = v0 - v_normal
        v_normal_length = wp.length(v_normal)
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        v_normal_new = -clamp_collide_elas * v_normal
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        min_pred = pred[neigh_indices[i]]
        min_dist = wp.length(min_pred - gt[i])
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        disable_backward=False,
    ):
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System")
        self.device = cfg.device

        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            )
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices
        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        self.controller_points = controller_points

        # Deal with the any collision detection
        self.object_collision_flag = 0
        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, dtype=torch.int32, device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )

            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], dtype=wp.float32, requires_grad=False
        )
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(self.wp_init_vertices, requires_grad=False)
            self.acc_count = wp.zeros(1, dtype=wp.int32, requires_grad=False)

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), dtype=wp.vec3, requires_grad=False
        )
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.num_valid_visibilities = int(self.gt_object_visibilities[1].sum())
            self.num_valid_motions = int(self.gt_object_motions_valid[0].sum())

            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), dtype=wp.vec3, requires_grad=False
            )
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), dtype=wp.vec3, requires_grad=False
            )

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(torch.tensor(spring_Y, dtype=torch.float32, device=self.device))
            * torch.ones(self.n_springs, dtype=torch.float32, device=self.device),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        self.collision_grid.build(self.wp_states[0].wp_x, self.collision_dist * 5.0)
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )

            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the vertive_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
                self.num_valid_visibilities,
                self.neigh_indices,
                cfg.chamfer_weight,
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )
```

qqtt/utils/__init__.py
```python
from .logger import logger
from .visualize import visualize_pc
from .config import cfg
```

qqtt/utils/config.py
```python
from .misc import singleton
import logging
import yaml
import pickle


@singleton
class Config:
    def __init__(self):
        self.data_type = "real"
        self.FPS = 30
        self.dt = 5e-5
        self.num_substeps = round(1.0 / self.FPS / self.dt)

        self.dashpot_damping = 100
        self.drag_damping = 3
        self.base_lr = 1e-3
        self.iterations = 250
        self.vis_interval = 10
        self.init_spring_Y = 3e3
        self.collide_elas = 0.5
        self.collide_fric = 0.3
        self.collide_object_elas = 0.7
        self.collide_object_fric = 0.3

        self.object_radius = 0.02
        self.object_max_neighbours = 30
        self.controller_radius = 0.04
        self.controller_max_neighbours = 50

        self.spring_Y_min = 0
        self.spring_Y_max = 1e5

        self.reverse_z = True
        self.vp_front = [1, 0, -2]
        self.vp_up = [0, 0, -1]
        self.vp_zoom = 1

        self.collision_dist = 0.06
        # Parameters on whether update the collision parameters
        self.collision_learn = True
        self.self_collision = False

        # DEBUG mode: set use_graph to False
        self.use_graph = True

        # Attribute for the real
        self.chamfer_weight = 1.0
        self.track_weight = 1.0
        self.acc_weight = 0.01

        # Other parameters for visualization
        self.overlay_path = None

        # track the last loaded pickle to avoid re-reading the file if the same
        # parameters are requested repeatedly.  This helps when zero-order and
        # first-order stages run in the same Python process.
        self._loaded_optimal_path: str | None = None

    def to_dict(self):
        # Convert the class to dictionary
        return {
            attr: getattr(self, attr)
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                setattr(self, key, value)

    def load_from_yaml(self, file_path):
        with open(file_path) as file:
            config_dict = yaml.safe_load(file)
        self.update_from_dict(config_dict)

    def load_from_yaml_with_optimal(
        self, yaml_path: str, optimal_path: str | None = None, use_global_spring_Y: bool = True
    ) -> None:
        """Load base parameters from YAML and optionally override with optimal pickle.

        Previously the calling scripts loaded the YAML and then loaded the
        ``optimal_params.pkl`` separately. When both zero-order and first-order
        stages were executed in sequence, ``optimal_params.pkl`` ended up being
        loaded twice.  This helper ensures that the file is loaded exactly once
        and clearly separates the two stages: pass ``optimal_path`` only for the
        first-order optimization step.
        """

        if optimal_path is None:
            # Zero-order stage
            self.load_zero_order_params(yaml_path)
            return

        # First-order stage
        self.load_first_order_params(
            yaml_path,
            optimal_path,
            use_global_spring_Y=use_global_spring_Y,
        )

    # ------------------------------------------------------------------
    # New helpers to emphasise the two-stage loading process
    # ------------------------------------------------------------------
    def load_zero_order_params(self, yaml_path: str) -> None:
        """Load parameters solely from ``yaml_path`` for zero-order CMA-ES.

        The method also resets ``_loaded_optimal_path`` so that the subsequent
        first-order optimization can safely call :func:`load_first_order_params`
        without thinking that the pickle has already been processed.
        """

        self.load_from_yaml(yaml_path)
        self._loaded_optimal_path = None

    def load_first_order_params(
        self,
        yaml_path: str,
        optimal_path: str,
        *,
        use_global_spring_Y: bool = True,
    ) -> None:
        """Load parameters for gradient-based refinement.

        ``optimal_path`` is cached to avoid redundant loading.  Previously the
        caller would load the YAML file and then re-apply ``optimal_params.pkl``
        each time, which reset ``init_spring_Y`` back to the YAML value before
        reading the pickle again.  By checking ``_loaded_optimal_path`` we ensure
        that repeated invocations are idempotent within a single process.
        """

        if optimal_path == self._loaded_optimal_path:
            # Avoid re-reading both YAML and pickle if nothing changed
            return

        self.load_from_yaml(yaml_path)

        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)

        self.set_optimal_params(optimal_params, use_global_spring_Y=use_global_spring_Y)
        self._loaded_optimal_path = optimal_path

        # Log the resulting spring stiffness for verification
        logging.info(f"Config init_spring_Y loaded: {self.init_spring_Y}")

    def set_optimal_params(self, optimal_params, use_global_spring_Y=True):
        if use_global_spring_Y:
            optimal_params["init_spring_Y"] = optimal_params.pop("global_spring_Y")
        else:
            optimal_params["init_spring_Y"] = self.init_spring_Y
        self.update_from_dict(optimal_params)


cfg = Config()
```

qqtt/utils/logger.py
```python
import logging
import os.path
import time
from typing import Optional

from .misc import singleton, master_only
from termcolor import colored
import sys


class Formatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    time_str = "%(asctime)s"
    level_str = "[%(levelname)7s]"
    msg_str = "%(message)s"
    file_str = "(%(filename)s:%(lineno)d)"

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class SteamFormatter(Formatter):

    FORMATS = {
        logging.DEBUG: colored(Formatter.msg_str, "cyan"),
        logging.INFO: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "white",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "white"),
        logging.WARNING: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "yellow",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "yellow"),
        logging.ERROR: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark"],
        )
        + colored(Formatter.msg_str, "red")
        + colored(" " + Formatter.file_str, "red", attrs=["dark"]),
        logging.CRITICAL: colored(
            " ".join([Formatter.time_str, Formatter.level_str, ""]),
            "red",
            attrs=["dark", "bold"],
        )
        + colored(
            Formatter.msg_str,
            "red",
            attrs=["bold"],
        )
        + colored(" " + Formatter.file_str, "red", attrs=["dark", "bold"]),
    }


class FileFormatter(Formatter):

    FORMATS = {
        logging.INFO: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.WARNING: " ".join(
            [Formatter.time_str, Formatter.level_str, Formatter.msg_str]
        ),
        logging.ERROR: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
        logging.CRITICAL: " ".join(
            [
                Formatter.time_str,
                Formatter.level_str,
                Formatter.msg_str,
                Formatter.file_str,
            ]
        ),
    }


@singleton
class ExpLogger(logging.Logger):

    def __init__(self, name: str | None = None):
        if name is None:
            name = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
        super().__init__(name)
        self.setLevel(logging.DEBUG)

        self.set_log_stream()
        self.filehandler = None

    @master_only
    def set_log_stream(self):
        self.stearmhandler = logging.StreamHandler()
        self.stearmhandler.setFormatter(SteamFormatter())
        self.stearmhandler.setLevel(logging.DEBUG)

        self.addHandler(self.stearmhandler)

    def remove_log_stream(self):
        self.removeHandler(self.stearmhandler)

    @master_only
    def set_log_file(self, path: str, name: str | None = None):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(
            path, f"{self.name}.log" if name is None else f"{name}.log"
        )
        self.filehandler = logging.FileHandler(file_path)
        self.filehandler.setFormatter(FileFormatter())
        self.filehandler.setLevel(logging.INFO)
        self.addHandler(self.filehandler)

    @master_only
    def info(self, msg, **kwargs) -> None:
        return super().info(msg, **kwargs)

    @master_only
    def warning(self, msg, **kwargs) -> None:
        return super().warning(msg, **kwargs)

    @master_only
    def error(self, msg, **kwargs) -> None:
        return super().error(msg, **kwargs)

    @master_only
    def debug(self, msg, **kwargs) -> None:
        return super().debug(msg, **kwargs)

    @master_only
    def critical(self, msg, **kwargs) -> None:
        return super().critical(msg, **kwargs)


logger = ExpLogger()

class StreamToLogger():
    def __init__(self, logger, log_level):
        super().__init__()
        self.logger = logger
        self.log_level = log_level

    def write(self, message):
        if message.strip():
            self.logger.log(self.log_level, message.strip())

    def flush(self):
        pass```

qqtt/utils/misc.py
```python
import functools
from torch import distributed as dist


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner
```

qqtt/utils/visualize.py
```python
import open3d as o3d
import numpy as np
import torch
import time
import cv2
from .config import cfg
import pyrender
import trimesh


def visualize_pc(
    object_points,
    object_colors=None,
    controller_points=None,
    object_visibilities=None,
    object_motions_valid=None,
    visualize=True,
    save_video=False,
    save_path=None,
    vis_cam_idx=0,
):
    # Deprecated function, use visualize_pc instead
    FPS = cfg.FPS
    width, height = cfg.WH
    intrinsic = cfg.intrinsics[vis_cam_idx]
    w2c = cfg.w2cs[vis_cam_idx]

    # Convert the stuffs to numpy if it's tensor
    if isinstance(object_points, torch.Tensor):
        object_points = object_points.cpu().numpy()
    if isinstance(object_colors, torch.Tensor):
        object_colors = object_colors.cpu().numpy()
    if isinstance(object_visibilities, torch.Tensor):
        object_visibilities = object_visibilities.cpu().numpy()
    if isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = object_motions_valid.cpu().numpy()
    if isinstance(controller_points, torch.Tensor):
        controller_points = controller_points.cpu().numpy()

    if object_colors is None:
        object_colors = np.tile(
            [1, 0, 0], (object_points.shape[0], object_points.shape[1], 1)
        )
    else:
        if object_colors.shape[1] < object_points.shape[1]:
            # If the object_colors is not the same as object_points, fill the colors with black
            object_colors = np.concatenate(
                [
                    object_colors,
                    np.ones(
                        (
                            object_colors.shape[0],
                            object_points.shape[1] - object_colors.shape[1],
                            3,
                        )
                    )
                    * 0.3,
                ],
                axis=1,
            )

    # The pcs is a 4d pcd numpy array with shape (n_frames, n_points, 3)
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visualize, width=width, height=height)

    if save_video and visualize:
        raise ValueError("Cannot save video and visualize at the same time.")

    # Initialize video writer if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(save_path, fourcc, FPS, (width, height))

    if controller_points is not None:
        controller_meshes = []
        prev_center = []
    for i in range(object_points.shape[0]):
        object_pcd = o3d.geometry.PointCloud()
        if object_visibilities is None:
            object_pcd.points = o3d.utility.Vector3dVector(object_points[i])
            object_pcd.colors = o3d.utility.Vector3dVector(object_colors[i])
        else:
            object_pcd.points = o3d.utility.Vector3dVector(
                object_points[i, np.where(object_visibilities[i])[0], :]
            )
            object_pcd.colors = o3d.utility.Vector3dVector(
                object_colors[i, np.where(object_visibilities[i])[0], :]
            )
        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            if controller_points is not None:
                # Use sphere mesh for each controller point
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    origin_color = [1, 0, 0]
                    controller_mesh = o3d.geometry.TriangleMesh.create_sphere(
                        radius=0.01
                    ).translate(origin)
                    controller_mesh.compute_vertex_normals()
                    controller_mesh.paint_uniform_color(origin_color)
                    controller_meshes.append(controller_mesh)
                    vis.add_geometry(controller_meshes[-1])
                    prev_center.append(origin)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            camera_params = o3d.camera.PinholeCameraParameters()
            intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
                width, height, intrinsic
            )
            camera_params.intrinsic = intrinsic_parameter
            camera_params.extrinsic = w2c
            view_control.convert_from_pinhole_camera_parameters(
                camera_params, allow_arbitrary=True
            )
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            if controller_points is not None:
                for j in range(controller_points.shape[1]):
                    origin = controller_points[i, j]
                    controller_meshes[j].translate(origin - prev_center[j])
                    vis.update_geometry(controller_meshes[j])
                    prev_center[j] = origin
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and write to video file if save_video is True
        if save_video:
            frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            frame = (frame * 255).astype(np.uint8)
            if cfg.overlay_path is not None:
                # Get the mask where the pixel is white
                mask = np.all(frame == [255, 255, 255], axis=-1)
                image_path = f"{cfg.overlay_path}/{vis_cam_idx}/{i}.png"
                overlay = cv2.imread(image_path)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                frame[mask] = overlay[mask]
            # Convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        if visualize:
            time.sleep(1 / FPS)

    vis.destroy_window()
    if save_video:
        video_writer.release()
```

script_inference.py
```python
import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"experiments/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    os.system(
        f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
    )
```

script_optimize.py
```python
import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    
    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    os.system(
        f"python optimize_cma.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )```

script_process_data.py
```python
import os
import csv

base_path = "./data/different_types"

os.system("rm -f timer.log")

with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        case_name = row[0]
        category = row[1]
        shape_prior = row[2]

        if not os.path.exists(f"{base_path}/{case_name}"):
            continue

        if shape_prior.lower() == "true":
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category} --shape_prior"
            )
        else:
            os.system(
                f"python process_data.py --base_path {base_path} --case_name {case_name} --category {category}"
            )
```

script_train.py
```python
import glob
import os
import json

base_path = "./data/different_types"
dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]

    # Read the train test split
    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)

    train_frame = split["train"][1]

    os.system(
        f"python train_warp.py --base_path {base_path} --case_name {case_name} --train_frame {train_frame}"
    )
```

tests/test_stiffness_loading.py
```python
import pickle
import importlib.util
import types
import sys

torch_stub = types.ModuleType("torch")
torch_stub.distributed = types.ModuleType("distributed")
sys.modules.setdefault("torch", torch_stub)

pkg = types.ModuleType("qqtt")
pkg.__path__ = ["qqtt"]
sys.modules["qqtt"] = pkg
utils_pkg = types.ModuleType("qqtt.utils")
utils_pkg.__path__ = ["qqtt/utils"]
sys.modules["qqtt.utils"] = utils_pkg

spec_misc = importlib.util.spec_from_file_location(
    "qqtt.utils.misc", "qqtt/utils/misc.py"
)
misc_mod = importlib.util.module_from_spec(spec_misc)
sys.modules["qqtt.utils.misc"] = misc_mod
spec_misc.loader.exec_module(misc_mod)

spec = importlib.util.spec_from_file_location(
    "qqtt.utils.config", "qqtt/utils/config.py"
)
config = importlib.util.module_from_spec(spec)
sys.modules["qqtt.utils.config"] = config
config.__package__ = "qqtt.utils"
spec.loader.exec_module(config)
cfg = config.cfg

def test_stiffness_loading(tmp_path):
    # zero-order stage: only load YAML
    cfg.load_zero_order_params('configs/real.yaml')
    yaml_stiffness = cfg.init_spring_Y
    assert isinstance(yaml_stiffness, float)

    # create a pickle with different stiffness
    new_val = yaml_stiffness + 10.0
    pkl_path = tmp_path / 'opt.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({'global_spring_Y': new_val}, f)

    # first-order stage: load YAML and override with pickle
    cfg.load_first_order_params('configs/real.yaml', str(pkl_path))
    assert cfg.init_spring_Y == new_val

    # loading again should not change the value
    cfg.load_first_order_params('configs/real.yaml', str(pkl_path))
    assert cfg.init_spring_Y == new_val
```

train_warp.py
```python
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
import os
import pickle
import json


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    # First-order stage loads YAML and then the optimal parameters produced by
    # the zero-order CMA-ES step.
    optimal_path = f"experiments_optimization/{case_name}/optimal_params.pkl"
    cfg.load_first_order_params(yaml_path, optimal_path)

    print(f"[DATA TYPE]: {cfg.data_type}")

    base_dir = f"experiments/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    trainer.train()
```

visualize_force.py
```python
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument("--case_name", type=str, default="double_lift_cloth_3")
    parser.add_argument("--n_ctrl_parts", type=int, default=2)
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    cfg.load_first_order_params(yaml_path, optimal_path)

    base_dir = f"./experiments/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.visualize_force(
        best_model_path, gaussians_path, args.n_ctrl_parts
    )```

visualize_material.py
```python
# Experimental feature to approximate the materials in the spring-mass model.
from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base_path",
        type=str,
        default="./data/different_types",
    )
    parser.add_argument(
        "--gaussian_path",
        type=str,
        default="./gaussian_output",
    )
    parser.add_argument("--case_name", type=str, default="double_stretch_sloth")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name

    if "cloth" in case_name or "package" in case_name:
        yaml_path = "configs/cloth.yaml"
    else:
        yaml_path = "configs/real.yaml"

    optimal_path = f"./experiments_optimization/{case_name}/optimal_params.pkl"
    logger.info(f"Load optimal parameters from: {optimal_path}")
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    cfg.load_first_order_params(yaml_path, optimal_path)

    base_dir = f"./experiments/{case_name}"

    # Set the intrinsic and extrinsic parameters for visualization
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"

    exp_name = "init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0"
    gaussians_path = f"{args.gaussian_path}/{case_name}/{exp_name}/point_cloud/iteration_10000/point_cloud.ply"

    logger.set_log_file(path=base_dir, name="inference_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    best_model_path = glob.glob(f"experiments/{case_name}/train/best_*.pth")[0]
    trainer.visualize_material(best_model_path, gaussians_path)
```

visualize_render_results.py
```python
import glob
import json
import numpy as np
import cv2

base_path = "./data/different_types"
prediction_dir = "./gaussian_output_dynamic_white"
human_mask_path = (
    "./data/different_types_human_mask"
)
object_mask_path = (
    "./data/render_eval_data"
)

height, width = 480, 848
FPS = 30
alpha = 0.7

dir_names = glob.glob(f"{base_path}/*")
for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    print(f"Processing {case_name}!!!!!!!!!!!!!!!")

    with open(f"{base_path}/{case_name}/split.json", "r") as f:
        split = json.load(f)
    frame_len = split["frame_len"]

    # Need to prepare the video
    for i in range(3):
        # Process each camera
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for .mp4 file format
        video_writer = cv2.VideoWriter(
            f"{prediction_dir}/{case_name}/{i}_integrate.mp4",
            fourcc,
            FPS,
            (width, height),
        )

        for frame_idx in range(frame_len):
            render_path = f"{prediction_dir}/{case_name}/{i}/{frame_idx:05d}.png"
            origin_image_path = f"{base_path}/{case_name}/color/{i}/{frame_idx}.png"
            human_mask_image_path = (
                f"{human_mask_path}/{case_name}/mask/{i}/0/{frame_idx}.png"
            )
            object_image_path = (
                f"{object_mask_path}/{case_name}/mask/{i}/{frame_idx}.png"
            )

            render_img = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
            origin_img = cv2.imread(origin_image_path)
            human_mask = cv2.imread(human_mask_image_path)
            human_mask = cv2.cvtColor(human_mask, cv2.COLOR_BGR2GRAY)
            human_mask = human_mask > 0
            object_mask = cv2.imread(object_image_path)
            object_mask = cv2.cvtColor(object_mask, cv2.COLOR_BGR2GRAY)
            object_mask = object_mask > 0

            final_image = origin_img.copy()
            render_mask = np.logical_and(
                (render_img != 0).any(axis=2), render_img[:, :, 3] > 100
            )
            render_img[~render_mask, 3] = 0

            final_image[:, :, :] = alpha * final_image + (1 - alpha) * np.array(
                [255, 255, 255], dtype=np.uint8
            )

            test_alpha = render_img[:, :, 3] / 255
            final_image[:, :, :] = render_img[:, :, :3] * test_alpha[
                :, :, None
            ] + final_image * (1 - test_alpha[:, :, None])

            final_image[human_mask] = alpha * origin_img[human_mask] + (
                1 - alpha
            ) * np.array([255, 255, 255], dtype=np.uint8)

            video_writer.write(final_image)

        video_writer.release()
```

