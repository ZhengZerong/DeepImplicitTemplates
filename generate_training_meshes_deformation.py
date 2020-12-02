#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import argparse
import json
import numpy as np
import os
import torch
import plyfile
import sys
from functools import partial
import deep_sdf
import deep_sdf.workspace as ws


class ThirdOrderInterpolater(object):
    def __init__(self, v_0, v_1, grad_0, grad_1):
        self.d = v_0
        self.c = grad_0
        self.a = (grad_1 - self.c) - 2 * (v_1 - self.c - self.d)
        self.b = v_1 - self.a - self.d - self.c

    def __call__(self, x):
        return self.a * x**3 + self.b * x**2 + self.c * x + self.d


def save_to_ply(verts, verts_warped, faces, ply_filename_out):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_color = 255 * (0.5 + 0.5 * verts_warped)

    verts_tuple = np.zeros(
        (num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "f4"), ("green", "f4"), ("blue", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = (verts[i][0], verts[i][1], verts[i][2],
                          verts_color[i][0], verts_color[i][1], verts_color[i][2])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def code_to_mesh(experiment_directory, checkpoint, step, start_id, end_id,
                 use_octree=True, resolution=256):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        # clamping_function = lambda x: x * specs["ClampingDistance"]
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
    latent_vectors = latent_vectors.cuda()

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    for i, latent_vector in enumerate(latent_vectors):
        if i < start_id:
            continue

        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("\\")
        instance_name = instance_name.split(".")[0]
        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_deformation_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )

        logging.info('Instance filename: ' + os.path.normpath(instance_filenames[i]))

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        offset = None
        scale = None

        # step_fn = ThirdOrderInterpolater(0.0, 1.0, 2.0, 1.0 / 8.0)
        step_fn = ThirdOrderInterpolater(0.0, 1.0, 1.0, 1.0)

        for s in range(step+1):
            decoder_func = partial(decoder.forward, step=step_fn(s/step))
            if use_octree:
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh_octree(
                        lambda x: decoder_func(x) - 0.005,
                        # decoder_func,
                        latent_vector,
                        mesh_filename + "_%03d" % s,
                        N=resolution,
                        max_batch=int(2 ** 17),
                        offset=offset,
                        scale=scale,
                        clamp_func=clamping_function
                    )
            else:
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh(
                        lambda x: decoder_func(x) - 0.005,
                        # decoder_func,
                        latent_vector,
                        mesh_filename + "_%03d" % s,
                        N=resolution,
                        max_batch=int(2 ** 17),
                        offset=offset,
                        scale=scale
                    )

            logging.info('Loading from: ' + mesh_filename + ("_%03d" % s) + ".ply")
            mesh = plyfile.PlyData.read(mesh_filename + ("_%03d" % s) + ".ply")
            mesh_v = []
            mesh_f = []
            for v in mesh.elements[0]:
                mesh_v.append(np.array((v[0], v[1], v[2])))
            for f in mesh.elements[1]:
                f = f[0]
                mesh_f.append(np.array([f[0], f[1], f[2]]))
            mesh_v = np.asarray(mesh_v)
            mesh_f = np.asarray(mesh_f)

            queries = torch.from_numpy(mesh_v).cuda()
            num_samples = queries.shape[0]
            latent_repeat = latent_vector.expand(num_samples, -1)
            inputs = torch.cat([latent_repeat, queries], 1)
            warped = []
            head = 0
            max_batch = int(2 ** 17)
            while head < num_samples:
                with torch.no_grad():
                    warped_, _ = decoder(
                        inputs[head: min(head + max_batch, num_samples)],
                        output_warped_points=True, step=step_fn(s / step))
                warped_ = warped_.detach().cpu().numpy()
                warped.append(warped_)
                head += max_batch
            warped = np.concatenate(warped, axis=0)

            save_to_ply(mesh_v, warped, mesh_f, mesh_filename + ("_%03d" % s) + "_color_coded.ply")
            # with open(mesh_filename + ("_%03d" % s) + "_color_coded.obj", "w") as fp:
            #     for v, vw in zip(mesh_v, warped):
            #         # c = s * (vw - t) + 0.5
            #         c = vw * 0.5 + 0.5
            #         fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
            #     for f in mesh_f:
            #         fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

        if i >= end_id:
            break


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )
    arg_parser.add_argument(
        "--start_id",
        dest="start_id",
        type=int,
        default=0,
        help="start_id.",
    )
    arg_parser.add_argument(
        "--end_id",
        dest="end_id",
        type=int,
        default=20,
        help="end_id.",
    )
    arg_parser.add_argument(
        "--step_num",
        dest="step_num",
        type=int,
        default=10,
        help="How many steps of deformation is generated",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )

    use_octree_group = arg_parser.add_mutually_exclusive_group()
    use_octree_group.add_argument(
        '--octree',
        dest='use_octree',
        action='store_true',
        help='Use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    use_octree_group.add_argument(
        '--no_octree',
        dest='use_octree',
        action='store_false',
        help='Don\'t use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    code_to_mesh(
        args.experiment_directory,
        args.checkpoint,
        args.step_num,
        args.start_id, args.end_id,
        args.use_octree,
        args.resolution)
