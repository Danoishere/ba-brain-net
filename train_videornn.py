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
import psutil
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
    num_frames = 18

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

    class_to_pos_net = net.ClassToPosNet().to(torchDevice)
    class_to_pos_net.load_state_dict(torch.load('active-models/posnet-model.mdl', map_location=torchDevice))
    class_to_pos_net.eval()

    pos_to_class_net = net.PosToClass(torchDevice).to(torchDevice)
    pos_to_class_net.load_state_dict(torch.load('active-models/colnet-model.mdl', map_location=torchDevice))
    pos_to_class_net.eval()

    uv_to_class_net = net.UVToClass(torchDevice).to(torchDevice)
    uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model.mdl', map_location=torchDevice))
    uv_to_class_net.eval()

    count_net = net.ObjCountNet(torchDevice).to(torchDevice)
    count_net.load_state_dict(torch.load('active-models/countnet-model.mdl', map_location=torchDevice))
    count_net.eval()

    has_below_above_net = net.HasObjectBelowAboveNet(torchDevice).to(torchDevice)
    has_below_above_net.load_state_dict(torch.load('active-models/classbelowabovenet-model.mdl', map_location=torchDevice))
    has_below_above_net.eval()

    q_net = net.QNet(torchDevice).to(torchDevice)
    q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.eval()

    class_below_above_net = net.ClassBelowAboveNet(torchDevice).to(torchDevice)
    class_below_above_net.load_state_dict(torch.load('active-models/neighbour-obj-model.mdl', map_location=torchDevice)) #TODO: activate when available
    class_below_above_net.eval()
    for m in [2]:
        episode = 0
        rl_episode = 0
        num_queries = config.num_queries
        skip = config.skip_factor

        eps = 1.0
        eps_min = 0.01
        eps_decay = 0.9999

        """
        plt.imshow(np.zeros((128,128,3)))
        plt.ion()
        plt.show()
        """
        success = [[] for i in list(range(num_frames))]
    
        tot_l2_err_class_to_pos = 0
        tot_acc_pos_to_class = 0
        tot_acc_uv_to_class = 0
        tot_sr_countnet = 0
        tot_acc_class_has_below_above = 0
        tot_loss_neighbour_obj = 0

        for i in range(1000):
            batch_x, scenes = queue.get()

            for repetition in range(1):
                frame = np.random.randint(0, sequence_length, batch_size)
                clip_length = num_frames
                cae.reset_hidden_state()
                
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
                    
                    encoded = cae(frame_input)
                    output = encoded.reshape(batch_size, -1)
                    v1_out = visual_cortex_net(output)
                    
                    reward = torch.zeros(batch_size, dtype=torch.float32).to(torchDevice)

                    # Sample 10 different objects combinations from each training batch.
                    #for i in range(num_queries):
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

                    if clip_frame == 17:
                        # Find position loss
                        s_objs = dict(scene_objects)
                        objs = count_net.infere(v1_out)
                        num_correct = 0
                        num_incorrect = 0
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

                        # Find position loss
                        y_pred_pos = class_to_pos_net(v1_out, y_col_oh, y_shape_oh)
                        tot_l2_err_class_to_pos += class_to_pos_net.loss(y_pred_pos, y_target_rel_pos_t).item()

                        # Find class loss
                        y_pred_col, y_pred_shape = pos_to_class_net(v1_out, y_target_rel_pos_t)
                        y_pred_col = torch.argmax(y_pred_col)
                        y_pred_shape = torch.argmax(y_pred_shape)

                        if y_pred_col == y_col_idx and y_pred_shape == y_shape_idx:
                            tot_acc_pos_to_class += 1.0

                        # UV to class loss
                        y_pred_col, y_pred_shape = uv_to_class_net(v1_out, y_uvs)
                        y_pred_col = torch.argmax(y_pred_col)
                        y_pred_shape = torch.argmax(y_pred_shape)

                        if y_pred_col == y_col_idx and y_pred_shape == y_shape_idx:
                            tot_acc_uv_to_class += 1.0

                        # Find hasAbove loss
                        y_pred_has_below_above = has_below_above_net(v1_out, y_col_oh, y_shape_oh)
                        y_pred_has_below_above = torch.argmax(y_pred_has_below_above)

                        if y_pred_has_below_above == y_has_below_above_idx:
                            tot_acc_class_has_below_above += 1.0

                        # Find class below above loss
                        y_pred_neighbour_obj_col, y_pred_neighbour_obj_shape = class_below_above_net(v1_out,y_col_oh, y_shape_oh)
                        y_pred_neighbour_obj_col = torch.argmax(y_pred_neighbour_obj_col)
                        y_pred_neighbour_obj_shape = torch.argmax(y_pred_neighbour_obj_shape)

                        if y_pred_neighbour_obj_col == y_neighbour_obj_col_idx and y_pred_neighbour_obj_shape == y_neighbour_obj_shape_idx:
                            tot_loss_neighbour_obj += 1.0

                        episode += 1

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
                    

    avg_l2_err_class_to_pos = tot_l2_err_class_to_pos/episode
    avg_acc_pos_to_class =  tot_acc_pos_to_class/episode
    avg_acc_uv_to_class =  tot_acc_uv_to_class/episode
    avg_sr_countnet =  tot_sr_countnet/episode
    avg_acc_class_has_below_above =  tot_acc_class_has_below_above/episode
    avg_acc_neighbour_obj =  tot_loss_neighbour_obj/episode

    print("L2-Error - Class to Pos:", avg_l2_err_class_to_pos)
    print("Accuracy - Pos to class:", avg_acc_pos_to_class)
    print("Accuracy - UV to class:", avg_acc_uv_to_class)
    print("Accuracy - Has class below/above:", avg_acc_class_has_below_above)
    print("Accuracy - Neighboring object:", avg_acc_neighbour_obj)
    print("Successrate - Enumeration stream:", avg_sr_countnet)
    
            
