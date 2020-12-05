#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh
import trimesh.sample
# from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def compute_trimesh_emd(gt_points, gen_mesh, offset, scale, num_mesh_samples=500):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)

    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)

    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices
    gt_points_np = np.random.permutation(gt_points_np)[:num_mesh_samples]

    # hist0 = hist1 = np.ones([num_mesh_samples], dtype=np.float64) / num_mesh_samples
    dist = np.linalg.norm(np.expand_dims(gt_points_np, axis=0) - np.expand_dims(gen_points_sampled, axis=1), axis=-1)
    # dist = dist.astype(np.float64)
    # emd = pyemd.emd(hist0, hist1, dist)
    assignment = linear_sum_assignment(dist)
    emd = dist[assignment].sum() / num_mesh_samples

    return emd
