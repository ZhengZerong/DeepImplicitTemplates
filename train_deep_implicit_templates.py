#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import numpy as np
import torch
import torch.utils.data as data_utils
import torch.nn
import signal
import sys
import os
import logging
import math
import json
import time
import datetime
import random

import deep_sdf
import deep_sdf.workspace as ws
from deep_sdf.lr_schedule import get_learning_rate_schedules
import deep_sdf.loss as loss


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def get_mean_latent_vector_magnitude(latent_vectors):
    return torch.mean(torch.norm(latent_vectors.weight.data.detach(), dim=1))


def append_parameter_magnitudes(param_mag_log, model):
    for name, param in model.named_parameters():
        if len(name) > 7 and name[:7] == "module.":
            name = name[7:]
        if name not in param_mag_log.keys():
            param_mag_log[name] = []
        param_mag_log[name].append(param.data.norm().item())


def apply_curriculum_l1_loss(pred_sdf_list, sdf_gt, loss_l1_soft, num_sdf_samples):
    soft_l1_eps_list = [2.5e-2, 1e-2, 2.5e-3, 0]
    soft_l1_lamb_list = [0, 0.1, 0.2, 0.5]
    sdf_loss = []
    for k in range(len(pred_sdf_list)):
        eps = soft_l1_eps_list[k]
        lamb = soft_l1_lamb_list[k]
        l = loss_l1_soft(pred_sdf_list[k], sdf_gt, eps=eps, lamb=lamb) / num_sdf_samples
        # l = loss_l1(pred_sdf_list[k], sdf_gt[i].cuda()) / num_sdf_samples
        sdf_loss.append(l)
    sdf_loss = sum(sdf_loss) / len(sdf_loss)
    return sdf_loss


def apply_pointwise_reg(warped_xyz_list, xyz_, huber_fn, num_sdf_samples):
    pw_loss = []
    for k in range(len(warped_xyz_list)):
        dist = torch.norm(warped_xyz_list[k] - xyz_, dim=-1)
        pw_loss.append(huber_fn(dist, delta=0.25) / num_sdf_samples)
        # pw_loss.append(torch.sum((warped_xyz_list[k] - xyz_) ** 2) / num_sdf_samples)
    pw_loss = sum(pw_loss) / len(pw_loss)
    return pw_loss


def apply_pointpair_reg(warped_xyz_list, xyz_, loss_lp, scene_per_split, num_sdf_samples):
    delta_xyz = warped_xyz_list[-1] - xyz_
    xyz_reshaped = xyz_.view((scene_per_split, -1, 3))
    delta_xyz_reshape = delta_xyz.view((scene_per_split, -1, 3))
    k = xyz_reshaped.shape[1] // 8
    lp_loss = torch.sum(loss_lp(
        xyz_reshaped[:, :k].view(scene_per_split, -1, 1, 3),
        xyz_reshaped[:, k:].view(scene_per_split, 1, -1, 3),
        delta_xyz_reshape[:, :k].view(scene_per_split, -1, 1, 3),
        delta_xyz_reshape[:, k:].view(scene_per_split, 1, -1, 3),
    )) / num_sdf_samples
    # lp_loss = torch.sum(
    #     loss_sm(xyz_, delta_xyz)
    # ) / num_sdf_samples
    return lp_loss


def main_function(experiment_directory, data_source, continue_from, batch_split):

    logging.info("running " + experiment_directory)

    # backup code
    now = datetime.datetime.now()
    code_bk_path = os.path.join(
        experiment_directory, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    ws.create_code_snapshot('./', code_bk_path,
                            extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh'),
                            exclude=('examples', 'third-party', 'bin'))

    specs = ws.load_experiment_specifications(experiment_directory)

    logging.info("Experiment description: \n" + specs["Description"])

    # data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    logging.info(specs["NetworkSpecs"])

    latent_size = specs["CodeLength"]

    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    lr_schedules = get_learning_rate_schedules(specs)

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))

    def save_latest(epoch):

        ws.save_model(experiment_directory, "latest.pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def save_checkpoints(epoch):

        ws.save_model(experiment_directory, str(epoch) + ".pth", decoder, epoch)
        ws.save_optimizer(experiment_directory, str(epoch) + ".pth", optimizer_all, epoch)
        ws.save_latent_vectors(experiment_directory, str(epoch) + ".pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        logging.info("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    def empirical_stat(latent_vecs, indices):
        lat_mat = torch.zeros(0).cuda()
        for ind in indices:
            lat_mat = torch.cat([lat_mat, latent_vecs[ind]], 0)
        mean = torch.mean(lat_mat, 0)
        var = torch.var(lat_mat, 0)
        return mean, var

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    scene_per_batch = specs["ScenesPerBatch"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    assert(scene_per_batch % batch_split == 0)  # requirements for computing chamfer loss
    scene_per_split = scene_per_batch // batch_split

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)

    code_bound = get_spec_with_default(specs, "CodeBound", None)

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()

    logging.info("training with {} GPU(s)".format(torch.cuda.device_count()))

    # if torch.cuda.device_count() > 1:
    decoder = torch.nn.DataParallel(decoder)

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = deep_sdf.data.SDFSamples(
        data_source, train_split, num_samp_per_scene, load_ram=True
    )

    if sdf_dataset.load_ram:
        num_data_loader_threads = 0
    else:
        num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    logging.debug("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )

    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    num_scenes = len(sdf_dataset)

    logging.info("There are {} scenes".format(num_scenes))

    logging.info(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    logging.debug(
        "initialized with mean magnitude {}".format(
            get_mean_latent_vector_magnitude(lat_vecs)
        )
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss_l1_soft = loss.SoftL1Loss(reduction="sum")
    loss_lp = torch.nn.DataParallel(loss.LipschitzLoss(k=0.5, reduction="sum"))
    huber_fn = loss.HuberFunc(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.module.warper.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": decoder.module.sdf_decoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ]
    )

    tensorboard_saver = ws.create_tensorboard_saver(experiment_directory)

    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    start_epoch = 1

    if continue_from is not None:
        if not os.path.exists(os.path.join(experiment_directory, ws.latent_codes_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.model_params_subdir, continue_from + ".pth")) or \
                not os.path.exists(os.path.join(experiment_directory, ws.optimizer_params_subdir, continue_from + ".pth")):
            logging.warning('"{}" does not exist! Ignoring this argument...'.format(continue_from))
        else:
            logging.info('continuing from "{}"'.format(continue_from))

            lat_epoch = ws.load_latent_vectors(
                experiment_directory, continue_from + ".pth", lat_vecs
            )

            model_epoch = ws.load_model_parameters(
                experiment_directory, continue_from, decoder
            )

            optimizer_epoch = ws.load_optimizer(
                experiment_directory, continue_from + ".pth", optimizer_all
            )

            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = ws.load_logs(
                experiment_directory
            )

            if not log_epoch == model_epoch:
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = ws.clip_logs(
                    loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, model_epoch
                )

            if not (model_epoch == optimizer_epoch and model_epoch == lat_epoch):
                raise RuntimeError(
                    "epoch mismatch: {} vs {} vs {} vs {}".format(
                        model_epoch, optimizer_epoch, lat_epoch, log_epoch
                    )
                )

            start_epoch = model_epoch + 1

            logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    logging.info(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )

    use_curriculum = get_spec_with_default(specs, "UseCurriculum", False)

    use_pointwise_loss = get_spec_with_default(specs, "UsePointwiseLoss", False)
    pointwise_loss_weight = get_spec_with_default(specs, "PointwiseLossWeight", 0.0)

    use_pointpair_loss = get_spec_with_default(specs, "UsePointpairLoss", False)
    pointpair_loss_weight = get_spec_with_default(specs, "PointpairLossWeight", 0.0)

    logging.info("pointwise_loss_weight = {}, pointpair_loss_weight = {}".format(
        pointwise_loss_weight, pointpair_loss_weight))

    for epoch in range(start_epoch, num_epochs + 1):

        start = time.time()

        logging.info("epoch {}...".format(epoch))

        decoder.train()

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        batch_num = len(sdf_loader)
        for bi, (sdf_data, indices) in enumerate(sdf_loader):

            # Process the input data
            sdf_data = sdf_data.reshape(-1, 4)

            num_sdf_samples = sdf_data.shape[0]

            sdf_data.requires_grad = False

            xyz = sdf_data[:, 0:3]
            sdf_gt = sdf_data[:, 3].unsqueeze(1)

            if enforce_minmax:
                sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            xyz = torch.chunk(xyz, batch_split)
            indices = torch.chunk(
                indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                batch_split,
            )

            sdf_gt = torch.chunk(sdf_gt, batch_split)

            batch_loss_sdf = 0.0
            batch_loss_pw = 0.0
            batch_loss_reg = 0.0
            batch_loss_pp = 0.0
            batch_loss = 0.0

            optimizer_all.zero_grad()

            for i in range(batch_split):

                batch_vecs = lat_vecs(indices[i])

                input = torch.cat([batch_vecs, xyz[i]], dim=1)
                xyz_ = xyz[i].cuda()

                # NN optimization
                warped_xyz_list, pred_sdf_list, _ = decoder(
                    input, output_warped_points=True, output_warping_param=True)

                if enforce_minmax:
                    # pred_sdf = pred_sdf * clamp_dist * 1.0
                    for k in range(len(pred_sdf_list)):
                        pred_sdf_list[k] = torch.clamp(pred_sdf_list[k], minT, maxT)

                if use_curriculum:
                    sdf_loss = apply_curriculum_l1_loss(
                        pred_sdf_list, sdf_gt[i].cuda(), loss_l1_soft, num_sdf_samples)
                else:
                    sdf_loss = loss_l1(pred_sdf_list[-1], sdf_gt[i].cuda()) / num_sdf_samples
                batch_loss_sdf += sdf_loss.item()
                chunk_loss = sdf_loss

                if do_code_regularization:
                    l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
                    reg_loss = l2_size_loss / num_sdf_samples
                    chunk_loss += code_reg_lambda * min(1.0, epoch / 100) * reg_loss.cuda()
                    batch_loss_reg += reg_loss.item()

                if use_pointwise_loss:
                    if use_curriculum:
                        pw_loss = apply_pointwise_reg(warped_xyz_list, xyz_, huber_fn, num_sdf_samples)
                    else:
                        pw_loss = apply_pointwise_reg(warped_xyz_list[-1:], xyz_, huber_fn, num_sdf_samples)
                    batch_loss_pw += pw_loss.item()
                    chunk_loss = chunk_loss + pw_loss.cuda() * pointwise_loss_weight * max(1.0, 10.0 * (1 - epoch / 100))

                if use_pointpair_loss:
                    if use_curriculum:
                        lp_loss = apply_pointpair_reg(warped_xyz_list, xyz_, loss_lp, scene_per_split, num_sdf_samples)
                    else:
                        lp_loss = apply_pointpair_reg(warped_xyz_list[-1:], xyz_, loss_lp, scene_per_split, num_sdf_samples)
                    batch_loss_pp += lp_loss.item()
                    chunk_loss += lp_loss.cuda() * pointpair_loss_weight * min(1.0, epoch / 100)

                chunk_loss.backward()
                batch_loss += chunk_loss.item()

            logging.debug("sdf_loss = {:.9f}, reg_loss = {:.9f}, pw_loss = {:.9f}, pp_loss = {:.9f}".format(
                batch_loss_sdf, batch_loss_reg, batch_loss_pw, batch_loss_pp))

            ws.save_tensorboard_logs(
                tensorboard_saver, epoch*batch_num + bi,
                loss_sdf=batch_loss_sdf, loss_pw=batch_loss_pw, loss_reg=batch_loss_reg,
                loss_pp=batch_loss_pp, loss_=batch_loss)

            loss_log.append(batch_loss)

            if grad_clip is not None:

                torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

            optimizer_all.step()

            # release memory
            del warped_xyz_list, pred_sdf_list, sdf_loss, pw_loss, \
                lp_loss, batch_loss_sdf, batch_loss_reg, batch_loss_pp, batch_loss_pw, batch_loss, chunk_loss

        end = time.time()

        seconds_elapsed = end - start
        timing_log.append(seconds_elapsed)

        lr_log.append([schedule.get_learning_rate(epoch) for schedule in lr_schedules])

        lat_mag_log.append(get_mean_latent_vector_magnitude(lat_vecs))

        append_parameter_magnitudes(param_mag_log, decoder)

        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            ws.save_logs(
                experiment_directory,
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )


if __name__ == "__main__":
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    import argparse

    arg_parser = argparse.ArgumentParser(description="Train a DeepSDF autodecoder")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.data_source, args.continue_from, int(args.batch_split))
