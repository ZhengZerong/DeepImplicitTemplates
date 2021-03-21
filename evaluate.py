#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import trimesh
from pathos.multiprocessing import ProcessPool as Pool

import deep_sdf
import deep_sdf.workspace as ws


def evaluate_one_instance(dataset, class_name, instance_name, experiment_directory, checkpoint, data_dir):
    logging.debug(
        "evaluating " + os.path.join(dataset, class_name, instance_name)
    )
    reconstructed_mesh_filename = ws.get_reconstructed_mesh_filename(
        experiment_directory, checkpoint, dataset, class_name, instance_name
    )

    logging.debug(
        'reconstructed mesh is "' + reconstructed_mesh_filename + '"'
    )

    if not os.path.isfile(reconstructed_mesh_filename):
        print('[WARNING] Skipping %s as it doesn\'t exists' % reconstructed_mesh_filename)
        return "", 0

    ground_truth_samples_filename = os.path.join(
        data_dir,
        "SurfaceSamples",
        dataset,
        class_name,
        instance_name + ".ply",
    )

    logging.debug(
        "ground truth samples are " + ground_truth_samples_filename
    )

    normalization_params_filename = os.path.join(
        data_dir,
        "NormalizationParameters",
        dataset,
        class_name,
        instance_name + ".npz",
    )

    logging.debug(
        "normalization params are " + ground_truth_samples_filename
    )

    ground_truth_points = trimesh.load(ground_truth_samples_filename)
    reconstruction = trimesh.load(reconstructed_mesh_filename)

    normalization_params = np.load(normalization_params_filename)

    chamfer_dist = deep_sdf.metrics.chamfer.compute_trimesh_chamfer(
        ground_truth_points,
        reconstruction,
        normalization_params["offset"],
        normalization_params["scale"],
    )

    earthmover_dist = deep_sdf.metrics.emd.compute_trimesh_emd(
        ground_truth_points,
        reconstruction,
        normalization_params["offset"],
        normalization_params["scale"],
    )

    logging.debug("chamfer distance: " + str(chamfer_dist))

    return os.path.join(dataset, class_name, instance_name), chamfer_dist, earthmover_dist


def evaluate(experiment_directory, checkpoint, data_dir, split_filename):

    with open(split_filename, "r") as f:
        split = json.load(f)

    chamfer_results = []
    p = Pool(8)
    ds = []
    cn = []
    inn = []
    exd = []
    ckp = []
    dtd = []

    print('data_preparing')
    for dataset in split:
        for class_name in split[dataset]:
            for iii, instance_name in enumerate(split[dataset][class_name]):
                ds.append(dataset)
                cn.append(class_name)
                inn.append(instance_name)
                exd.append(experiment_directory)
                ckp.append(checkpoint)
                dtd.append(data_dir)

    print('multi thread start')
    chamfer_results = p.map(evaluate_one_instance, ds, cn, inn, exd, ckp, dtd)
    print(np.mean([q[1] for q in chamfer_results]), np.median([q[1] for q in chamfer_results]))
    print(np.mean([q[2] for q in chamfer_results]), np.median([q[2] for q in chamfer_results]))

    with open(
        os.path.join(
            ws.get_evaluation_dir(experiment_directory, checkpoint, True), "chamfer_and_emd.csv"
        ),
        "w",
    ) as f:
        f.write("shape, chamfer_dist\n")
        for result in chamfer_results:
            f.write("{}, {}, {}\n".format(result[0], result[1], result[2]))


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Evaluate a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include experiment specifications in "
        + '"specs.json", and logging will be done in this directory as well.',
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint to test.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--split",
        "-s",
        dest="split_filename",
        required=True,
        help="The split to evaluate.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    evaluate(
        args.experiment_directory,
        args.checkpoint,
        args.data_source,
        args.split_filename,
    )
