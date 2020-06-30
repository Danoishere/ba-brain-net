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
import matplotlib
from random import shuffle, randint, choice
from autoencoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import config
from matplotlib.ticker import MaxNLocator
import gc
import sys

plt.style.use('seaborn')

def action_idx_to_action(indices):
    actions = []
    for idx in indices:
        actions.append(config.actions[idx])

    return np.array(actions,dtype=np.int)

def train_video_rnn(queue, lock, torchDevice, load_model=True):

    eval = {}
    """
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
    })
    """
    

    plt.rcParams['figure.figsize'] = (7,3.6)

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
    cae.eval()

    lgn_net = net.VisionNet().to(torchDevice)
    lgn_net.load_state_dict(torch.load('active-models/lgn-net.mdl', map_location=torchDevice))
    lgn_net.eval()

    visual_cortex_net = net.VisualCortexNet().to(torchDevice)
    visual_cortex_net.load_state_dict(torch.load('active-models/visual-cortex-net.mdl', map_location=torchDevice))
    visual_cortex_net.eval()

    uv_to_class_net = net.UVToClass(torchDevice).to(torchDevice)
    uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model.mdl', map_location=torchDevice))
    uv_to_class_net.eval()

    count_net = net.ObjCountNet(torchDevice).to(torchDevice)
    count_net.load_state_dict(torch.load('active-models/countnet-model.mdl', map_location=torchDevice))
    count_net.eval()

    q_net = net.QNet(torchDevice).to(torchDevice)
    q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.eval()

    pos_is_ripe_net = net.PosIsRipeClass(torchDevice).to(torchDevice)
    pos_is_ripe_net.load_state_dict(torch.load('active-models/pos_is_ripe_model.mdl', map_location=torchDevice))
    pos_is_ripe_net.eval()

    for m in [2]:
        episode = 0
        rl_episode = 0
        num_queries = config.num_queries

        tot_acc_pos_to_ripe = 0
        tot_acc_uv_to_class = 0
        tot_sr_countnet = 0

        for i in range(20):
            batch_x, scenes = queue.get()

            frame = np.random.randint(0, sequence_length, batch_size)
            clip_length = 9
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

                y_target_rel_pos = []
                y_target_has_below_above = []
                
                all_uvs = []
                obj_ripe = []
                all_objs = []
                all_cam_pos = []


                scene = scenes[0]
                scene_objects = scene["objects"]
                rnd_obj = np.random.choice(list(scene_objects.keys()))

                last_frame_transf_mat = np.array(scene["cam_base_matricies"][frame[0]])
                last_frame_transf_mat_inv = np.linalg.inv(last_frame_transf_mat)

                last_frame_uv = scene["ss_objs"][frame[0]][rnd_obj]
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

                if clip_frame == (clip_length - 1):
                    # Find position loss
                    s_objs = dict(scene_objects)
                    #objs = count_net.infere(v1_out)
                    num_correct = 0
                    num_incorrect = 0
                    """
                    print('-------------------------')
                    for found_obj in objs:
                        key = found_obj[1] + '-' + found_obj[0]
                        if key in s_objs:
                            print("Found:", key)
                            num_correct += 1
                            del s_objs[key]
                        else:
                            num_incorrect += 1
                            print("Failed:", key)

                    for remaining_obj in s_objs:
                        num_incorrect += 1
                        print("Not counted:", remaining_obj)
                    print('-------------------------')
                    print("correctness:", num_correct/(num_correct + num_incorrect))
                    print('-------------------------')

                    tot_sr_countnet += num_correct/(num_correct + num_incorrect)
                    """

                    # Find position loss
                    # UV to class loss
                    y_pred_is_ripe_uv = uv_to_class_net(v1_out, y_uvs)
                    y_pred_is_ripe_uv = torch.sigmoid(y_pred_is_ripe_uv)
                    y_pred_is_ripe_uv_bool = y_pred_is_ripe_uv.item() > 0.5 

                    if y_pred_is_ripe_uv_bool == is_ripe_bool:
                        tot_acc_uv_to_class += 1.0


                    # Find class is ripe loss
                    y_pred_is_ripe_pos = pos_is_ripe_net(v1_out, y_target_rel_pos_t)
                    y_pred_is_ripe_pos = torch.sigmoid(y_pred_is_ripe_pos)
                    y_pred_is_ripe_pos_bool = y_pred_is_ripe_pos.item() > 0.5

                    if y_pred_is_ripe_pos_bool == is_ripe_bool:
                        tot_acc_pos_to_ripe += 1.0
                    
                    episode += 1

                    avg_acc_pos_to_ripe =  tot_acc_pos_to_ripe/episode
                    avg_acc_uv_to_class =  tot_acc_uv_to_class/episode
                    avg_sr_countnet =  tot_sr_countnet/episode

                    print("Accuracy - Pos to ripe:", avg_acc_pos_to_ripe)
                    print("Accuracy - UV to ripe:", avg_acc_uv_to_class)
                    print("Successrate - Enumeration stream:", avg_sr_countnet)

                if first_action_taken:
                    memory.append((q_net_out, action_idx, reward))

                q_net_out = q_net(v1_out)
                first_action_taken = True

                if m == 0:
                    action_idx = torch.argmax(q_net_out,dim=1).cpu().numpy()
                elif m == 1:
                    action_idx = np.array([4])
                elif m == 2:
                    action_idx = np.array([5])
                elif m == 3:
                    action_idx = np.array([6])
                else:
                    action_idx = np.array([1]) * randint(0, 6)

                print('Actions', action_idx)

                """
                
                """
                    

    avg_acc_pos_to_ripe =  tot_acc_pos_to_ripe/episode
    avg_acc_uv_to_class =  tot_acc_uv_to_class/episode
    avg_sr_countnet =  tot_sr_countnet/episode

    print("Accuracy - Pos to ripe:", avg_acc_pos_to_ripe)
    print("Accuracy - UV to ripe:", avg_acc_uv_to_class)
    print("Successrate - Enumeration stream:", avg_sr_countnet)
    
            
