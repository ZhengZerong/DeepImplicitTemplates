#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch

import deep_sdf.utils


def create_mesh(
    decoder, latent_vec, filename, N=256, max_batch=(32 ** 3 * 4), offset=None, scale=None, volume_size=2.0
):
    start = time.time()
    ply_filename = filename

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-volume_size/2.0, -volume_size/2.0, -volume_size/2.0]
    voxel_size = volume_size / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 4)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() // N) % N
    samples[:, 0] = ((overall_index.long() // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    samples.requires_grad = False

    head = 0

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        samples[head : min(head + max_batch, num_samples), 3] = (
            deep_sdf.utils.decode_sdf(decoder, latent_vec, sample_subset)
            .squeeze(1)
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    logging.debug("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def create_mesh_octree(
        decoder, latent_vec, filename, N=256, max_batch=32 ** 3, offset=None, scale=None, clamp_func=None,
        volume_size=2.0):
    start = time.time()
    ply_filename = filename

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-volume_size / 2.0, -volume_size / 2.0, -volume_size / 2.0]
    voxel_size = volume_size / (N - 1)

    overall_index = np.arange(0, N ** 3)
    samples = np.zeros([N ** 3, 4], dtype=np.float32)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index // N) % N
    samples[:, 0] = ((overall_index // N) // N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples = samples.reshape([N, N, N, 4])

    sdf_values = np.zeros([N, N, N], dtype=np.float32)
    dirty = np.ones([N, N, N], dtype=np.bool)
    grid_mask = np.zeros_like(dirty, dtype=np.bool)

    init_res = 64
    ignore_thres = volume_size / N / 4
    reso = N // init_res
    while reso > 0:
        grid_mask[0:N:reso, 0:N:reso, 0:N:reso] = True

        test_mask = np.logical_and(grid_mask, dirty)
        samples_ = samples[test_mask]
        samples_ = torch.from_numpy(samples_).cuda()
        sdf_ = []

        head = 0
        print(samples_.shape[0])
        while head < samples_.shape[0]:
            query_idx = torch.arange(head, min(head + max_batch, samples_.shape[0])).long().cuda()
            s = (deep_sdf.utils.decode_sdf(
                    decoder, latent_vec, samples_[query_idx, :3]).view([-1]).detach()
            )
            if clamp_func is not None:
                s = clamp_func(s)

            sdf_.append(s.cpu().numpy())
            head += max_batch

        sdf_values[test_mask] = np.concatenate(sdf_, axis=-1)

        if reso <= 1:
            break

        N_ds = N // reso - 1
        overall_index_ds = np.arange(0, N_ds ** 3)
        samples_ds = np.zeros([N_ds ** 3, 4], dtype=np.int32)

        # transform first 3 columns
        # to be the x, y, z index
        samples_ds[:, 2] = overall_index_ds % N_ds
        samples_ds[:, 1] = (overall_index_ds // N_ds) % N_ds
        samples_ds[:, 0] = ((overall_index_ds // N_ds) // N_ds) % N_ds
        samples_ds *= reso

        dirty_ds = dirty[samples_ds[:, 0] + reso // 2,
                         samples_ds[:, 1] + reso // 2, samples_ds[:, 2] + reso // 2]
        samples_ds = samples_ds[dirty_ds]
        v0 = sdf_values[samples_ds[:, 0], samples_ds[:, 1], samples_ds[:, 2]]
        v1 = sdf_values[samples_ds[:, 0], samples_ds[:, 1], samples_ds[:, 2] + reso]
        v2 = sdf_values[samples_ds[:, 0], samples_ds[:, 1] + reso, samples_ds[:, 2]]
        v3 = sdf_values[samples_ds[:, 0], samples_ds[:, 1] + reso, samples_ds[:, 2] + reso]
        v4 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1], samples_ds[:, 2]]
        v5 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1], samples_ds[:, 2] + reso]
        v6 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1] + reso, samples_ds[:, 2]]
        v7 = sdf_values[samples_ds[:, 0] + reso, samples_ds[:, 1] + reso, samples_ds[:, 2] + reso]

        vs = np.asarray([v0, v1, v2, v3, v4, v5, v6, v7])
        vmn = np.min(vs, axis=0)
        vmx = np.max(vs, axis=0)
        v_ = 0.5 *(vmx + vmn)
        clean_flag = (vmx - vmn) < ignore_thres
        for sample, v in zip(samples_ds[clean_flag], v_[clean_flag]):
            x, y, z = sample[0], sample[1], sample[2]
            sdf_values[x:x+reso, y:y+reso, z:z+reso] = v
            dirty[x:x + reso, y:y + reso, z:z + reso] = False

        reso //= 2

    end = time.time()
    logging.debug("sampling takes: %f" % (end - start))

    convert_sdf_samples_to_ply(
        sdf_values,
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    input_3d_sdf_array,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    if isinstance(input_3d_sdf_array, torch.Tensor):
        numpy_3d_sdf_tensor = input_3d_sdf_array.numpy()
    elif isinstance(input_3d_sdf_array, np.ndarray):
        numpy_3d_sdf_tensor = input_3d_sdf_array
    else:
        raise NotImplementedError

    verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
