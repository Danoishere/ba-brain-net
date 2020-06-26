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

    uv_to_class_net = net.UVToClass(torchDevice).to(torchDevice)
    #uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model.mdl', map_location=torchDevice))
    uv_to_class_net.train()

    count_net = net.ObjCountNet(torchDevice).to(torchDevice)
    #count_net.load_state_dict(torch.load('active-models/countnet-model.mdl', map_location=torchDevice))
    count_net.train()

    q_net = net.QNet(torchDevice).to(torchDevice)
    q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.train()

    pos_is_ripe_net = net.PosIsRipeClass(torchDevice).to(torchDevice)
    # pos_is_ripe_net-load_state_dict(torch.load('active-models/pos_is_ripe_model.mdl', map_location=torchDevice))
    pos_is_ripe_net.train()

    params = []
    params += list(cae.parameters())
    params += list(lgn_net.parameters())
    params += list(visual_cortex_net.parameters())
    params += list(uv_to_class_net.parameters())
    params += list(count_net.parameters())
    params += list(q_net.parameters())
    params += list(pos_is_ripe_net())

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

                frame_input = torch.tensor(frame_input, requires_grad=True, dtype=torch.float32).to(torchDevice)
                
                encoded = cae.encode(frame_input)
                output = encoded.reshape(batch_size, -1)
                output = lgn_net(output)
                v1_out = visual_cortex_net(output)
                
                reward = torch.zeros(batch_size, dtype=torch.float32).to(torchDevice)
                
                tot_loss_is_ripe = []
                tot_loss_uv_to_class = []
                tot_loss_countnet = []

                # Sample 10 different objects combinations from each training batch.
                for i in range(num_queries):
                    # Predict color, based on location
                    y_target_rel_pos = []
                    y_target_has_below_above = []
                    
                    all_uvs = []
                    obj_ripe = []
                    all_objs = []
                    all_cam_pos = []

                    for scene_idx in range(len(scenes)):
                        scene = scenes[scene_idx]
                        scene_objects = scene["objects"]
                        rnd_obj = np.random.choice(list(scene_objects.keys()))

                        last_frame_transf_mat = np.array(scene["cam_base_matricies"][frame[scene_idx]])
                        last_frame_transf_mat_inv = np.linalg.inv(last_frame_transf_mat)

                        last_frame_uv = scene["ss_objs"][frame[scene_idx]][rnd_obj]
                        all_uvs.append([last_frame_uv["screen_x"], last_frame_uv["screen_y"]])
                        
                        obj_pos = scene_objects[rnd_obj]['pos']
                        obj_rel_pos = last_frame_transf_mat_inv @ np.array(obj_pos + [1])
                        obj_rel_pos = obj_rel_pos[:3]

                        is_ripe_bool = scene_objects[rnd_obj]['is_ripe'] #change that
                        binary_ripe = 0.0
                        if is_ripe_bool:
                            binary_ripe = 1.0

                        #y_target_pos.append(obj_pos)
                        y_target_rel_pos.append(obj_rel_pos)
                        obj_ripe.append(binary_ripe)

                    # oh = one-hot
                    y_ripe_bin = torch.tensor(obj_ripe, requires_grad=True, dtype=torch.float32).to(torchDevice)
                    y_uvs = torch.tensor(all_uvs, requires_grad=True, dtype=torch.float32).to(torchDevice)
                    y_target_rel_pos_t = torch.tensor(y_target_rel_pos, requires_grad=True, dtype=torch.float32).to(torchDevice)

                    # UV to class loss
                    y_pred_ripe = uv_to_class_net(v1_out, y_uvs)
                    tot_loss_uv_to_class += [uv_to_class_net.loss(y_pred_ripe, y_ripe_bin)]

                    # Find class is ripe loss
                    y_pred_pos_is_ripe = pos_is_ripe_net(v1_out, y_target_rel_pos_t)
                    tot_loss_is_ripe += [pos_is_ripe_net.loss(y_pred_pos_is_ripe, y_ripe_bin)]

                """
                print(obj_rel_pos)
                print(rnd_obj)
                print(obj_col_oh)
                print(obj_shape_oh)
                img = np.moveaxis(batch_x[last_frame, scenes.index(scene), :3, :, :], 0,2)
                plt.imshow(img)
                plt.show()
                """

                p_dones, p_ripe, p_pos = count_net(v1_out)
                tot_loss_countnet = count_net.loss(p_dones, p_ripe, p_pos, scenes, frame)

                tot_loss_is_ripe = torch.stack(tot_loss_is_ripe)
                tot_loss_is_ripe = torch.mean(tot_loss_is_ripe,dim=0)

                tot_loss_pos_to_class = torch.stack(tot_loss_pos_to_class)
                tot_loss_pos_to_class = torch.mean(tot_loss_pos_to_class,dim=0)

                tot_loss_uv_to_class = torch.stack(tot_loss_uv_to_class)
                tot_loss_uv_to_class = torch.mean(tot_loss_uv_to_class,dim=0)

                tot_loss_class_has_below_above = torch.stack(tot_loss_class_has_below_above)
                tot_loss_class_has_below_above = torch.mean(tot_loss_class_has_below_above,dim=0)
                
                tot_loss_neighbour_obj = torch.stack(tot_loss_neighbour_obj)
                tot_loss_neighbour_obj = torch.mean(tot_loss_neighbour_obj)

                print('Episode', episode,', Clip Frame', clip_frame,'Action', action_idx, ', Loss Pos.:', torch.mean(tot_loss_is_ripe).item(), ", Eps.", eps)

                tot_loss_sum =  tot_loss_is_ripe + \
                                tot_loss_pos_to_class + \
                                tot_loss_uv_to_class + \
                                tot_loss_countnet + \
                                tot_loss_class_has_below_above + \
                                tot_loss_neighbour_obj


                if first_loss_initialized:
                    current_loss = tot_loss_sum.clone().detach().float()
                    reward += (last_loss - current_loss).detach()
                
                last_loss = tot_loss_sum.clone().detach().float()
                first_loss_initialized = True

                loss += torch.mean(tot_loss_sum)
                
                writer.add_scalar("Loss/Class-to-Position-Loss", torch.mean(tot_loss_is_ripe).item(), episode)
                writer.add_scalar("Loss/Position-to-Class-Loss", torch.mean(tot_loss_pos_to_class).item(), episode)
                writer.add_scalar("Loss/UV-to-Class-Loss", torch.mean(tot_loss_uv_to_class).item(), episode)
                writer.add_scalar("Loss/Obj-Count-Loss", torch.mean(tot_loss_countnet).item(), episode)
                writer.add_scalar("Loss/Has-Below-Above-Loss", torch.mean(tot_loss_class_has_below_above).item(), episode)
                writer.add_scalar("Loss/Class-Below-Above-Loss", torch.mean(tot_loss_neighbour_obj).item(), episode)

                if episode % 500 == 0:
                    torch.save(lgn_net.state_dict(), 'active-models/lgn-net.mdl')
                    torch.save(visual_cortex_net.state_dict(), 'active-models/visual-cortex-net.mdl')
                    torch.save(uv_to_class_net.state_dict(), 'active-models/uvtoclass-model.mdl')
                    torch.save(cae.state_dict(), 'active-models/cae-model.mdl')
                    torch.save(count_net.state_dict(), 'active-models/countnet-model.mdl')
                    torch.save(q_net.state_dict(), 'active-models/q-net-model.mdl')
                    torch.save(pos_is_ripe_net.state_dict(), 'activate-models/pos_is_ripe_model.mdl')
                    torch.save(optimizer.state_dict(), 'active-models/optimizer.opt')
                    

                episode += 1

                if first_action_taken:
                    memory.append((q_net_out, action_idx, reward))

                q_net_out = q_net(v1_out)
                first_action_taken = True
                action_idx = torch.argmax(q_net_out,dim=1).cpu().numpy()

                for scene_idx in range(len(action_idx)):
                    if eps > np.random.random():
                        action_idx[scene_idx] = randint(0, len(config.actions) - 1)
                
            eps *= eps_decay
            eps = max([eps_min, eps])

            rl_loss = []
            for i in range(len(memory)):
                rl_episode += 1
                mem = memory[i]
                q_values = mem[0]
                action_idx = mem[1]
                reward = mem[2].clone()

                print(q_values)
                discount = 1
                for r in range(i + 1, len(memory)):
                    future_mem = memory[r]
                    reward += 0.99**discount * future_mem[2]
                    discount += 1

                q_loss = q_net.loss(action_idx, q_values, reward)
                writer.add_scalar("Loss/Q-Net-Loss", q_loss.item(), rl_episode)
                rl_loss += [q_loss]

            rl_loss = torch.mean(torch.stack(rl_loss))

            loss += rl_loss
            loss.backward()
            optimizer.step()


            
            
