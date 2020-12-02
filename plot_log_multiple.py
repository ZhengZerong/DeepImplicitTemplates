#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import deep_sdf
import deep_sdf.workspace as ws


def running_mean(x, N):
    x = np.asarray(x)
    x_sorted = np.sort(x)
    k = x_sorted[len(x)-200]
    x[x > k] = k
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def load_logs(experiment_directories, type):
    fig, ax = plt.subplots()

    for ei, experiment_directory in enumerate(experiment_directories):

        logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

        logging.info("latest epoch is {}".format(logs["epoch"]))

        num_iters = len(logs["loss"])
        iters_per_epoch = num_iters / logs["epoch"]
        smoothed_loss_41 = running_mean(logs["loss"], 41)

        if type == "loss":

            ax.plot(
                np.arange(20, num_iters - 20) / iters_per_epoch,
                smoothed_loss_41,
                label=experiment_directory)

            ax.set(xlabel="Epoch", ylabel="Loss", title="Training Loss")
        else:
            raise Exception('unrecognized plot type "{}"'.format(type))

    ax.grid()
    ax.legend()
    plt.show()


if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser(description="Plot DeepSDF training logs")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        nargs="+",
        help="The experiment directory. This directory should include experiment "
        + "specifications in 'specs.json', and logging will be done in this directory "
        + "as well",
    )
    arg_parser.add_argument("--type", "-t", dest="type", default="loss")

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    load_logs(args.experiment_directory, args.type)
