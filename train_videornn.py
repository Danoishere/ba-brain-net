
import zipfile
import glob, os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import net
from net import Query
import matplotlib.pyplot as plt
from random import shuffle
from autoencoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



def train_video_rnn(queue, lock, load_model=True):

    now = datetime.now()
    current_time = now.strftime("-%H-%M-%S")
    writer = SummaryWriter('tensorboard/train' + current_time, flush_secs=10)

    lr=0.0001
    batch_size = 32
    sequence_length = 36
    w, h = 128, 128
    colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
    shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]

    cae = ConvAutoencoder().cuda()
    cae.load_state_dict(torch.load('cae-model-csipo.mdl'))
    cae.train()

    vnet = net.Net().cuda()
    vnet.load_state_dict(torch.load('vnet-model-csipo.mdl'))
    vnet.train()

    posnet = net.PosNet().cuda()
    posnet.load_state_dict(torch.load('posnet-model-csipo.mdl'))
    posnet.train()

    colnet = net.ColNet().cuda()
    colnet.load_state_dict(torch.load('colnet-model-csipo.mdl'))
    colnet.train()
    
    
    # cae.requires_grad = False

    col_citerion = nn.CrossEntropyLoss().cuda()
    pos_criterion = nn.MSELoss().cuda()

    params = []
    params += list(cae.parameters())
    params += list(vnet.parameters())
    params += list(colnet.parameters())
    params += list(posnet.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    losses = []
    episodes = []

    episode = 0
    num_queries = 16
    start_train_frames = 2048
    while True:
        batch_x, scenes = queue.get()

        optimizer.zero_grad()
        vnet.init_hidden()

        skip = 4
        # Pass batch frame by frame
        for frame in range(int(sequence_length/skip)):
            # Dimensionen batch_x
            # (frame-nr (36), batch-nr, color, height, width)
            frame_input = torch.tensor(batch_x[int(frame*skip)], requires_grad=True).float().cuda()
            encoded = cae.encode(frame_input)
            encoded = encoded.reshape(batch_size, -1)
            output = vnet(encoded)

        tot_loss_pos = torch.tensor(0.0).cuda()
        tot_loss_col = torch.tensor(0.0).cuda()

        # Sample 10 different objects combinations from each training batch.
        for i in range(num_queries):
            batch_y_is_green = np.zeros((batch_size))
            batch_y_col = np.zeros((batch_size))

            # Predict color, based on location
            y_target_pos = []
            obj_col_onehots = []
            obj_col_indices = []
            obj_shape_onehots = []
            all_objs = []
            all_cam_pos = []

            for scene in scenes:
                scene_objects = scene["objects"]
                rnd_obj = np.random.choice(list(scene_objects.keys()))
                obj_pos = scene_objects[rnd_obj]['pos']
                obj_col_idx = colors.index(scene_objects[rnd_obj]['color-name'])
                obj_shape_idx = shapes.index(rnd_obj.split("-")[0])

                obj_col_oh = np.zeros(len(colors))
                obj_col_oh[obj_col_idx] = 1.0
                obj_shape_oh = np.zeros(len(shapes))
                obj_shape_oh[obj_shape_idx] = 1.0

                y_target_pos.append(obj_pos)
                obj_col_onehots.append(obj_col_oh)
                obj_shape_onehots.append(obj_shape_oh)
                obj_col_indices.append(obj_col_idx)

                scene_cam_pos = []
                for cam_pos_frame in scene["cam_positions"]:
                    scene_cam_pos.append(cam_pos_frame)

                all_cam_pos.append(scene_cam_pos)


            x_all_cam_pos = np.asarray(all_cam_pos)
            x_all_cam_pos = np.swapaxes(x_all_cam_pos, 0,1)

            # oh = one-hot
            y_col_oh = torch.tensor(obj_col_onehots, requires_grad=True, dtype=torch.float32).cuda()
            y_shape_oh = torch.tensor(obj_shape_onehots, requires_grad=True, dtype=torch.float32).cuda()
            y_target_pos_t = torch.tensor(y_target_pos).cuda()

            # Find position loss
            y_pred_pos = posnet(output, y_shape_oh, y_col_oh)
            loss_pos = euclidean_distance_loss(y_pred_pos, y_target_pos_t)
            tot_loss_pos += loss_pos

            # Find color loss
            y_col_idx = torch.tensor(obj_col_indices, dtype=torch.long).cuda()
            y_pred_col = colnet(output, y_target_pos_t)
            loss_col = col_citerion(y_pred_col, y_col_idx)

            tot_loss_col += loss_col

        print('Episode', episode, ', Loss Pos.:', tot_loss_pos.item()/num_queries, ', Loss Col.:', tot_loss_col.item()/num_queries)

        loss = tot_loss_col + tot_loss_pos

        loss.backward()
        nn.utils.clip_grad_norm_(params, 0.07)
        optimizer.step()
        
        episodes.append(episode)
        losses.append(loss.item()/num_queries)

        writer.add_scalar("Loss/Pos. Loss", tot_loss_pos.item()/num_queries, episode)
        writer.add_scalar("Loss/Col. Loss", tot_loss_col.item()/num_queries, episode)
        episode += 1

        if episode % 1000 == 0:
            torch.save(vnet.state_dict(), 'vnet-model-csipo.mdl')
            torch.save(posnet.state_dict(), 'posnet-model-csipo.mdl')
            torch.save(colnet.state_dict(), 'colnet-model-csipo.mdl')
            torch.save(cae.state_dict(), 'cae-model-csipo.mdl')


    plt.plot(episodes, losses)
    plt.show()

def euclidean_distance_loss(y_pred_pos, y_target_pos_t):


    # sqrt(x^2 + y^2 + z^2)

    diff = y_pred_pos - y_target_pos_t
    diff_squared = diff**2
    diff_sum = torch.sum(diff_squared, dim=1)
    diff_sum_sqrt = torch.sqrt(diff_sum)

    loss_pos = torch.mean(diff_sum_sqrt)
    return loss_pos
    
