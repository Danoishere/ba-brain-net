
import zipfile
import glob, os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import net
from master_net import MasterNet
from net import Query
import matplotlib.pyplot as plt
from random import shuffle, randint
from autoencoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import config


def train_video_rnn(queue, lock, torchDevice, load_model=True):

    now = datetime.now()
    current_time = now.strftime("-%H-%M-%S")
    writer = SummaryWriter('tensorboard/train' + current_time, flush_secs=10)

    lr=config.lr
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    w, h = config.w, config.h

    masternet = MasterNet(torchDevice)

    episode = 0
    while True:
        last_frame = 0
        batch_x, scenes = queue.get()

        for repetition in range(3):
            masternet.train_on_batch(batch_x, scenes, masternet.perform_update)