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
from random import shuffle, randint, choice
from autoencoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import config

import gc
import sys

def action_idx_to_action(indices):
    actions = []
    for idx in indices:
        actions.append(config.actions[idx])

    return np.array(actions,dtype=np.int)

def train_video_rnn(queue, lock, torchDevice, load_model=True):
    now = datetime.now()
    current_time = now.strftime("-%H-%M-%S")
    writer = SummaryWriter('tensorboard/train' + current_time, flush_secs=10)

    lr=config.lr
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    w, h = config.w, config.h

    colors = config.colors
    colors_n = config.colors_n
    shapes = config.shapes
    shapes_n = config.shapes_n
    belowAbove = config.belowAbove

    cae = ConvAutoencoder().to(torchDevice)
    cae.load_state_dict(torch.load('active-models/cae-model.mdl', map_location=torchDevice))
    cae.train()

    lgn_net = net.VisionNet().to(torchDevice)
    lgn_net.load_state_dict(torch.load('active-models/lgn-net.mdl', map_location=torchDevice))
    lgn_net.train()

    visual_cortex_net = net.VisualCortexNet().to(torchDevice)
    visual_cortex_net.load_state_dict(torch.load('active-models/visual-cortex-net.mdl', map_location=torchDevice))
    visual_cortex_net.train()

    has_red_net = net.ContainsRedObject(torchDevice).to(torchDevice)
    #has_red_net.load_state_dict(torch.load('active-models/red-net.mdl', map_location=torchDevice))
    has_red_net.train()

    q_net = net.QNet(torchDevice).to(torchDevice)
    q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.train()

    params = []
    params += list(has_red_net.parameters())

    optimizer = torch.optim.RMSprop(params, lr=lr) #.Adam(params, lr=lr)

    episode = 0
    rl_episode = 0
    num_queries = config.num_queries
    skip = config.skip_factor

    eps = 0.5
    eps_min = 0.01
    eps_decay = 0.9999
    
    while True:
        batch_x, scenes = queue.get()

        for repetition in range(3):
            frame = np.random.randint(0, sequence_length, batch_size)
            clip_length = 9
            optimizer.zero_grad()
            lgn_net.init_hidden(torchDevice)
            
            first_loss_initialized = False
            clip_frame = 0
            action_idx = np.ones(batch_size, dtype=np.int)*3
            memory = []
            first_action_taken = False
            
            loss = torch.tensor(0.0, dtype=torch.float32).to(torchDevice)
            for step in range(clip_length):
                clip_frame += 1
                frame += action_idx_to_action(action_idx)
                frame = frame % sequence_length

                frame_input = np.zeros((batch_size, 4, config.w, config.h))
                for i in range(batch_size):
                    frame_input[i] = batch_x[frame[i],i]

                frame_input = torch.tensor(frame_input, dtype=torch.float32).to(torchDevice)
                
                with torch.no_grad():
                    encoded = cae.encode(frame_input)
                    output = encoded.reshape(batch_size, -1)
                    output = lgn_net(output)
                    v1_out = visual_cortex_net(output)

                tot_loss_has_red = []
                all_has_reds = []

                for scene_idx in range(len(scenes)):
                    scene = scenes[scene_idx]
                    scene_objects = scene["objects"]
                    has_red = 0.0
                    for o in scene_objects:
                        if o.split('-')[1] == 'red':
                            has_red = 1.0
                    all_has_reds.append(has_red)

                # oh = one-hot
                y_has_red = torch.tensor(all_has_reds, dtype=torch.float32).to(torchDevice)
                
                v1_out = v1_out.clone().detach().requires_grad_(False)
                y_pred_has_red = has_red_net(v1_out)
                tot_loss_has_red += [has_red_net.loss(y_pred_has_red, y_has_red)]

                """
                print(obj_rel_pos)
                print(rnd_obj)
                print(obj_col_oh)
                print(obj_shape_oh)
                img = np.moveaxis(batch_x[last_frame, scenes.index(scene), :3, :, :], 0,2)
                plt.imshow(img)
                plt.show()
                """

                tot_loss_has_red = torch.stack(tot_loss_has_red)
                tot_loss_has_red = torch.mean(tot_loss_has_red,dim=0)

                print('Episode', episode,', Clip Frame', clip_frame,'Action', action_idx, ', Loss Pos.:', torch.mean(tot_loss_has_red).item(), ", Eps.", eps)

                loss =  tot_loss_has_red
                loss = torch.mean(loss)
                loss.backward()
                optimizer.step()
                
                writer.add_scalar("Loss/Has-Red-Loss", torch.mean(loss).item(), episode)
                if episode % 500 == 0:
                    torch.save(has_red_net.state_dict(), 'active-models/red-net.mdl')

                episode += 1

                q_net_out = q_net(v1_out)
                first_action_taken = True
                action_idx = torch.argmax(q_net_out,dim=1).cpu().numpy()
            


            
            
