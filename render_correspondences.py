import numpy as np
import trimesh
import plyfile
from pyrender import IntrinsicsCamera, DirectionalLight, Mesh, Scene, Viewer


def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def load_ply_data(mesh_path):
    mesh = plyfile.PlyData.read(mesh_path)
    mesh_v = []
    mesh_vc = []
    mesh_f = []
    for v in mesh.elements[0]:
        mesh_v.append(np.array((v[0], v[1], v[2])))
        mesh_vc.append(np.array((v[3], v[4], v[5])))
    for f in mesh.elements[1]:
        f = f[0]
        mesh_f.append(np.array([f[0], f[1], f[2]]))
    mesh_v = np.asarray(mesh_v)
    mesh_f = np.asarray(mesh_f)
    mesh_vc = np.asarray(mesh_vc) / 255.0
    return mesh_v, mesh_vc, mesh_f


def main(mesh_path):
    # rendering conf
    ambient_light = 0.8
    directional_light = 1.0
    img_res = 512
    cam_f = 500
    cam_c = img_res / 2.0

    scene = Scene(ambient_light=np.array([ambient_light, ambient_light, ambient_light, 1.0]))

    mesh_v, mesh_vc, mesh_f = load_ply_data(mesh_path)
    mesh_ = trimesh.Trimesh(vertices=mesh_v, faces=mesh_f, vertex_colors=mesh_vc)
    points_mesh = Mesh.from_trimesh(mesh_, smooth=True, material=None)
    mesh_node = scene.add(points_mesh)

    cam = IntrinsicsCamera(fx=cam_f, fy=cam_f, cx=cam_c, cy=cam_c)
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_1 = scene.add(direc_l, pose=np.matmul(rotationy(30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_2 = scene.add(direc_l, pose=np.matmul(rotationy(-30), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=directional_light)
    light_node_3 = scene.add(direc_l, pose=np.matmul(rotationy(-180), rotationx(45)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=(directional_light-0.5))
    light_node_4 = scene.add(direc_l, pose=np.matmul(rotationy(0), rotationx(-10)))

    ################
    # rendering
    cam_node = scene.add(cam, pose=cam_pose)
    render_flags = {
        'flip_wireframe': False,
        'all_wireframe': False,
        'all_solid': False,
        'shadows': True,
        'vertex_normals': False,
        'face_normals': False,
        'cull_faces': True,
        'point_size': 1.0,
    }
    viewer_flags = {
        'mouse_pressed': False,
        'rotate': False,
        'rotate_rate': np.pi / 6.0,
        'rotate_axis': np.array([0.0, 1.0, 0.0]),
        'view_center': np.array([0.0, 0.0, 0.0]),
        'record': False,
        'use_raymond_lighting': False,
        'use_direct_lighting': False,
        'lighting_intensity': 3.0,
        'use_perspective_cam': True,
        'window_title': 'DIT',
        'refresh_rate': 25.0,
        'fullscreen': False,
        'show_world_axis': False,
        'show_mesh_axes': False,
        'caption': None,
        'save_one_frame': False,
    }
    v = Viewer(scene, viewport_size=(512, 512), render_flags=render_flags,
                    viewer_flags=viewer_flags, run_in_thread=False)
    v.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', required=True, type=str)
    args = parser.parse_args()
    main(args.input_path)
