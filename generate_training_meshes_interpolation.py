#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import numpy as np
import os
import torch
import sys
import deep_sdf
import deep_sdf.workspace as ws


def code_to_mesh(experiment_directory, checkpoint, step_num, start_id, end_id,
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
    latent_vectors_1 = latent_vectors[:-1]
    latent_vectors_2 = latent_vectors[1:]

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_filenames = deep_sdf.data.get_instance_filenames(data_source, train_split)

    for i, (latent_vector_1, latent_vector_2) in enumerate(zip(latent_vectors_1, latent_vectors_2)):
        if i < start_id:
            continue

        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("\\")
        instance_name = instance_name.split(".")[0]

        print("{} {} {}".format(dataset_name, class_name, instance_name))

        mesh_dir = os.path.join(
            experiment_directory,
            ws.interpolation_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        for s in range(step_num+1):

            mesh_filename = os.path.join(mesh_dir, "%04d" % i)

            print(mesh_filename)

            latent_vector = latent_vector_1 * (1.0 - s/ step_num) + latent_vector_2 * (s / step_num)

            offset = None
            scale = None

            if use_octree:
                with torch.no_grad():
                    deep_sdf.mesh.create_mesh_octree(
                        decoder,
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
                        decoder,
                        # decoder_func,
                        latent_vector,
                        mesh_filename + "_%03d" % s,
                        N=resolution,
                        max_batch=int(2 ** 17),
                        offset=offset,
                        scale=scale
                    )
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
