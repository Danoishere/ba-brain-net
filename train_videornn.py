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

    class_to_pos_net = net.ClassToPosNet().to(torchDevice)
    class_to_pos_net.load_state_dict(torch.load('active-models/posnet-model.mdl', map_location=torchDevice))
    class_to_pos_net.train()

    pos_to_class_net = net.PosToClass(torchDevice).to(torchDevice)
    pos_to_class_net.load_state_dict(torch.load('active-models/colnet-model.mdl', map_location=torchDevice))
    pos_to_class_net.train()

    uv_to_class_net = net.UVToClass(torchDevice).to(torchDevice)
    uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model.mdl', map_location=torchDevice))
    uv_to_class_net.train()

    count_net = net.ObjCountNet(torchDevice).to(torchDevice)
    count_net.load_state_dict(torch.load('active-models/countnet-model.mdl', map_location=torchDevice))
    count_net.train()

    has_below_above_net = net.HasObjectBelowAboveNet(torchDevice).to(torchDevice)
    has_below_above_net.load_state_dict(torch.load('active-models/classbelowabovenet-model.mdl', map_location=torchDevice))
    has_below_above_net.train()

    q_net = net.QNet(torchDevice).to(torchDevice)
    q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.train()

    class_below_above_net = net.ClassBelowAboveNet(torchDevice).to(torchDevice)
    class_below_above_net.load_state_dict(torch.load('active-models/neighbour-obj-model.mdl', map_location=torchDevice)) #TODO: activate when available
    class_below_above_net.train()

    params = []
    params += list(cae.parameters())
    params += list(lgn_net.parameters())
    params += list(visual_cortex_net.parameters())
    params += list(class_to_pos_net.parameters())
    params += list(pos_to_class_net.parameters())
    params += list(uv_to_class_net.parameters())
    params += list(count_net.parameters())
    params += list(has_below_above_net.parameters())
    params += list(q_net.parameters())
    params += list(class_below_above_net.parameters())

    optimizer = torch.optim.RMSprop(params, lr=lr) #.Adam(params, lr=lr)

    episode = 0
    rl_episode = 0
    num_queries = config.num_queries
    skip = config.skip_factor

    eps = 1.0
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
                
                tot_loss_class_to_pos = []
                tot_loss_pos_to_class = []
                tot_loss_uv_to_class = []
                tot_loss_countnet = []
                tot_loss_class_has_below_above = []
                tot_loss_neighbour_obj = []

                # Sample 10 different objects combinations from each training batch.
                for i in range(num_queries):
                    # Predict color, based on location
                    y_target_rel_pos = []
                    y_target_has_below_above = []
                    
                    all_uvs = []
                    obj_col_onehots = []
                    obj_col_indices = []
                    obj_shape_indices = []
                    obj_shape_onehots = []
                    below_above_indices = []
                    neighbour_obj_col_indices = []
                    neighbour_obj_shape_indices = []
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

                        obj_col_idx = colors.index(scene_objects[rnd_obj]['color-name'])
                        obj_shape_idx = shapes.index(rnd_obj.split("-")[0])

                        obj_has_above = 'is_below' in scene_objects[rnd_obj].keys() #is below -> has above
                        obj_has_below = 'is_above' in scene_objects[rnd_obj].keys() #is above -> has below

                        neighbour_obj = ''
                        if obj_has_below:
                            below_above_idx = belowAbove.index("below")
                            neighbour_obj = scene_objects[rnd_obj]['is_above']
                        elif obj_has_above:
                            below_above_idx = belowAbove.index("above")
                            neighbour_obj = scene_objects[rnd_obj]['is_below']
                        else:
                            below_above_idx = belowAbove.index("standalone")
                            neighbour_obj = "None-none" #no neighbour obj

                        neighbour_obj_shape = neighbour_obj.split("-")[0]
                        neighbour_obj_color = neighbour_obj.split("-")[1]
                        neighbour_obj_shape_idx = shapes_n.index(neighbour_obj_shape)
                        neighbour_obj_col_idx = colors_n.index(neighbour_obj_color)

                        obj_col_oh = np.zeros(len(colors))
                        obj_col_oh[obj_col_idx] = 1.0
                        obj_shape_oh = np.zeros(len(shapes))
                        obj_shape_oh[obj_shape_idx] = 1.0

                        #y_target_pos.append(obj_pos)
                        y_target_rel_pos.append(obj_rel_pos)
                        obj_col_onehots.append(obj_col_oh)
                        obj_shape_onehots.append(obj_shape_oh)
                        obj_col_indices.append(obj_col_idx)
                        obj_shape_indices.append(obj_shape_idx)
                        below_above_indices.append(below_above_idx)
                        neighbour_obj_shape_indices.append(neighbour_obj_shape_idx)
                        neighbour_obj_col_indices.append(neighbour_obj_col_idx)

                    # oh = one-hot
                    y_col_oh = torch.tensor(obj_col_onehots, requires_grad=True, dtype=torch.float32).to(torchDevice)
                    y_shape_oh = torch.tensor(obj_shape_onehots, requires_grad=True, dtype=torch.float32).to(torchDevice)
                    y_uvs = torch.tensor(all_uvs, requires_grad=True, dtype=torch.float32).to(torchDevice)
                    y_target_rel_pos_t = torch.tensor(y_target_rel_pos, requires_grad=True, dtype=torch.float32).to(torchDevice)

                    y_col_idx = torch.tensor(obj_col_indices, dtype=torch.long).to(torchDevice)
                    y_shape_idx = torch.tensor(obj_shape_indices, dtype=torch.long).to(torchDevice)
                    
                    y_has_below_above_idx = torch.tensor(below_above_indices, dtype=torch.long).to(torchDevice)
                    y_neighbour_obj_shape_idx = torch.tensor(neighbour_obj_shape_indices, dtype=torch.long).to(torchDevice)
                    y_neighbour_obj_col_idx = torch.tensor(neighbour_obj_col_indices, dtype=torch.long).to(torchDevice)

                    # Find position loss
                    y_pred_pos = class_to_pos_net(v1_out, y_col_oh, y_shape_oh)
                    tot_loss_class_to_pos += [class_to_pos_net.loss(y_pred_pos, y_target_rel_pos_t)]

                    # Find class loss
                    y_pred_col, y_pred_shape = pos_to_class_net(v1_out, y_target_rel_pos_t)
                    tot_loss_pos_to_class += [pos_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)]

                    # UV to class loss
                    y_pred_col, y_pred_shape = uv_to_class_net(v1_out, y_uvs)
                    tot_loss_uv_to_class += [uv_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)]

                    # Find hasAbove loss
                    y_pred_has_below_above = has_below_above_net(v1_out, y_col_oh, y_shape_oh)
                    tot_loss_class_has_below_above += [has_below_above_net.loss(y_pred_has_below_above,y_has_below_above_idx)]

                    # Find class below above loss
                    y_pred_neighbour_obj_col, y_pred_neighbour_obj_shape = class_below_above_net(v1_out,y_col_oh, y_shape_oh)
                    tot_loss_neighbour_obj += [class_below_above_net.loss(y_pred_neighbour_obj_col, y_pred_neighbour_obj_shape, y_neighbour_obj_col_idx, y_neighbour_obj_shape_idx)]


                """
                print(obj_rel_pos)
                print(rnd_obj)
                print(obj_col_oh)
                print(obj_shape_oh)
                img = np.moveaxis(batch_x[last_frame, scenes.index(scene), :3, :, :], 0,2)
                plt.imshow(img)
                plt.show()
                """

                p_dones, p_cols, p_shapes, p_pos = count_net(v1_out)
                tot_loss_countnet = count_net.loss(p_dones, p_cols, p_shapes, p_pos, scenes, frame)

                tot_loss_class_to_pos = torch.stack(tot_loss_class_to_pos)
                tot_loss_class_to_pos = torch.mean(tot_loss_class_to_pos,dim=0)

                tot_loss_pos_to_class = torch.stack(tot_loss_pos_to_class)
                tot_loss_pos_to_class = torch.mean(tot_loss_pos_to_class,dim=0)

                tot_loss_uv_to_class = torch.stack(tot_loss_uv_to_class)
                tot_loss_uv_to_class = torch.mean(tot_loss_uv_to_class,dim=0)

                tot_loss_class_has_below_above = torch.stack(tot_loss_class_has_below_above)
                tot_loss_class_has_below_above = torch.mean(tot_loss_class_has_below_above,dim=0)
                
                tot_loss_neighbour_obj = torch.stack(tot_loss_neighbour_obj)
                tot_loss_neighbour_obj = torch.mean(tot_loss_neighbour_obj)

                print('Episode', episode,', Clip Frame', clip_frame,'Action', action_idx, ', Loss Pos.:', torch.mean(tot_loss_class_to_pos).item(), ", Eps.", eps)

                tot_loss_sum =  tot_loss_class_to_pos + \
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
                
                writer.add_scalar("Loss/Class-to-Position-Loss", torch.mean(tot_loss_class_to_pos).item(), episode)
                writer.add_scalar("Loss/Position-to-Class-Loss", torch.mean(tot_loss_pos_to_class).item(), episode)
                writer.add_scalar("Loss/UV-to-Class-Loss", torch.mean(tot_loss_uv_to_class).item(), episode)
                writer.add_scalar("Loss/Obj-Count-Loss", torch.mean(tot_loss_countnet).item(), episode)
                writer.add_scalar("Loss/Has-Below-Above-Loss", torch.mean(tot_loss_class_has_below_above).item(), episode)
                writer.add_scalar("Loss/Class-Below-Above-Loss", torch.mean(tot_loss_neighbour_obj).item(), episode)

                if episode % 500 == 0:
                    torch.save(lgn_net.state_dict(), 'active-models/lgn-net.mdl')
                    torch.save(visual_cortex_net.state_dict(), 'active-models/visual-cortex-net.mdl')
                    torch.save(class_to_pos_net.state_dict(), 'active-models/posnet-model.mdl')
                    torch.save(pos_to_class_net.state_dict(), 'active-models/colnet-model.mdl')
                    torch.save(uv_to_class_net.state_dict(), 'active-models/uvtoclass-model.mdl')
                    torch.save(cae.state_dict(), 'active-models/cae-model.mdl')
                    torch.save(count_net.state_dict(), 'active-models/countnet-model.mdl')
                    torch.save(has_below_above_net.state_dict(), 'active-models/classbelowabovenet-model.mdl')
                    torch.save(class_below_above_net.state_dict(), 'active-models/neighbour-obj-model.mdl')
                    torch.save(q_net.state_dict(), 'active-models/q-net-model.mdl')
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


            
            
