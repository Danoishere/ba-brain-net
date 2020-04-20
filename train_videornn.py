
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
from random import shuffle, randint
from autoencoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import config


def train_video_rnn(queue, lock, load_model=True):

    now = datetime.now()
    current_time = now.strftime("-%H-%M-%S")
    writer = SummaryWriter('tensorboard/train' + current_time, flush_secs=10)

    lr=config.lr
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    w, h = config.w, config.h

    colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
    shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]

    cae = ConvAutoencoder().cuda()
    cae.load_state_dict(torch.load('active-models/cae-model-csipo.mdl'))
    cae.train()

    vision_net = net.VisionNet().cuda()
    vision_net.load_state_dict(torch.load('active-models/vnet-model-csipo.mdl'))
    vision_net.train()

    ventral_net = net.VentralNet().cuda()
    #ventral_net.load_state_dict(torch.load('active-models/ventral-net-csipo.mdl'))
    ventral_net.train()

    dorsal_net = net.DorsalNet().cuda()
    #dorsal_net.load_state_dict(torch.load('active-models/dorsal-net-csipo.mdl'))
    dorsal_net.train()

    class_to_pos_net = net.ClassToPosNet().cuda()
    #class_to_pos_net.load_state_dict(torch.load('active-models/posnet-model-csipo.mdl'))
    class_to_pos_net.train()

    pos_to_class_net = net.PosToClass().cuda()
    #pos_to_class_net.load_state_dict(torch.load('active-models/colnet-model-csipo.mdl'))
    pos_to_class_net.train()

    uv_to_class_net = net.UVToClass().cuda()
    #uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model-csipo.mdl'))
    uv_to_class_net.train()

    params = []
    params += list(cae.parameters())
    params += list(vision_net.parameters())

    params += list(ventral_net.parameters())
    params += list(dorsal_net.parameters())

    params += list(class_to_pos_net.parameters())
    # params += list(pos_to_class_net.parameters())
    #params += list(uv_to_class_net.parameters())
    

    optimizer = torch.optim.Adam(params, lr=lr)
    losses = []
    episodes = []

    episode = 0
    num_queries = config.num_queries
    skip = config.skip_factor

    while True:
        batch_x, scenes = queue.get()
        last_frame = 0
        # Pass batch frame by frame

        to_frame = int(sequence_length/skip) - randint(0,int((sequence_length/skip)/4))
        selected_frame = randint(0,35)
        frames = list(range(0, sequence_length))
        shuffle(frames)
        
        # for frame in range(to_frame):
        for frame in frames: #elected_frame, selected_frame + 1):

            optimizer.zero_grad()
            vision_net.init_hidden()

            # Dimensionen batch_x
            # (frame-nr (36), batch-nr, color, height, width)
            #frame_input = torch.tensor(batch_x[int(frame*skip)], requires_grad=True).float().cuda()
            frame_input = torch.tensor(batch_x[frame], requires_grad=True).float().cuda()
            encoded = cae.encode(frame_input)
            output = encoded.reshape(batch_size, -1)
            # output = vision_net(encoded)
            last_frame = frame

            vent_out = ventral_net(output)
            dors_out = dorsal_net(output)

            tot_loss_class_to_pos = torch.tensor(0.0).cuda()
            tot_loss_pos_to_class = torch.tensor(0.0).cuda()
            tot_loss_uv_to_class = torch.tensor(0.0).cuda()


            # Sample 10 different objects combinations from each training batch.
            for i in range(num_queries):
                batch_y_is_green = np.zeros((batch_size))
                batch_y_col = np.zeros((batch_size))

                # Predict color, based on location
                # y_target_pos = []
                y_target_rel_pos = []
                
                all_uvs = []
                obj_col_onehots = []
                obj_col_indices = []
                obj_shape_indices = []
                obj_shape_onehots = []
                all_objs = []
                all_cam_pos = []

                for scene in scenes:
                    scene_objects = scene["objects"]
                    rnd_obj = np.random.choice(list(scene_objects.keys()))

                    last_frame_transf_mat = np.array(scene["cam_base_matricies"][last_frame])
                    last_frame_transf_mat_inv = np.linalg.inv(last_frame_transf_mat)

                    last_frame_uv = scene["ss_objs"][last_frame][rnd_obj]
                    all_uvs.append([last_frame_uv["screen_x"], last_frame_uv["screen_y"]])
                    
                    obj_pos = scene_objects[rnd_obj]['pos']
                    obj_rel_pos = last_frame_transf_mat_inv @ np.array(obj_pos + [1])
                    obj_rel_pos = obj_rel_pos[:3]

                    

                    obj_col_idx = colors.index(scene_objects[rnd_obj]['color-name'])
                    obj_shape_idx = shapes.index(rnd_obj.split("-")[0])

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

                    """
                    print(obj_rel_pos)
                    print(rnd_obj)
                    print(obj_col_oh)
                    print(obj_shape_oh)
                    img = np.moveaxis(batch_x[last_frame, scenes.index(scene), :3, :, :], 0,2)
                    plt.imshow(img)
                    plt.show()
                    """


                # oh = one-hot
                y_col_oh = torch.tensor(obj_col_onehots, requires_grad=True, dtype=torch.float32).cuda()
                y_shape_oh = torch.tensor(obj_shape_onehots, requires_grad=True, dtype=torch.float32).cuda()
                y_target_rel_pos_t = torch.tensor(y_target_rel_pos, dtype=torch.float32).cuda()
                y_col_idx = torch.tensor(obj_col_indices, dtype=torch.long).cuda()
                y_shape_idx = torch.tensor(obj_shape_indices, dtype=torch.long).cuda()
                y_uvs = torch.tensor(all_uvs, requires_grad=True, dtype=torch.float32).cuda()

                # Find position loss
                y_pred_pos = class_to_pos_net(vent_out, dors_out, y_col_oh, y_shape_oh)
                tot_loss_class_to_pos += class_to_pos_net.loss(y_pred_pos, y_target_rel_pos_t)

                # Find class loss
                y_pred_col, y_pred_shape = pos_to_class_net(vent_out, dors_out, y_target_rel_pos_t)
                tot_loss_pos_to_class += pos_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)

                # UV to class loss
                y_pred_col, y_pred_shape = uv_to_class_net(vent_out, dors_out, y_uvs)
                tot_loss_uv_to_class += uv_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)

            print('Episode', episode, 
                ', Loss Pos.:', tot_loss_class_to_pos.item()/num_queries, 
                ', Pos to Class Loss:', tot_loss_pos_to_class.item()/num_queries,
                ', UV to Class Loss:', tot_loss_uv_to_class.item()/num_queries)

            loss_tot = tot_loss_pos_to_class.item() + tot_loss_class_to_pos.item() + tot_loss_uv_to_class.item()

            loss = (tot_loss_pos_to_class.item()/loss_tot) * tot_loss_pos_to_class + \
                (tot_loss_class_to_pos.item()/loss_tot) * tot_loss_class_to_pos + \
                (tot_loss_uv_to_class.item()/loss_tot) * tot_loss_uv_to_class

            (tot_loss_class_to_pos).backward()
            nn.utils.clip_grad_value_(params, 0.1)
            optimizer.step()
            
            episodes.append(episode)
            losses.append(loss.item()/num_queries)

            writer.add_scalar("Loss/Class-to-Position-Loss", tot_loss_class_to_pos.item()/num_queries, episode)
            writer.add_scalar("Loss/Position-to-Class-Loss", tot_loss_pos_to_class.item()/num_queries, episode)
            writer.add_scalar("Loss/UV-to-Class-Loss", tot_loss_uv_to_class.item()/num_queries, episode)
            episode += 1

            if episode % 500 == 0:
                torch.save(vision_net.state_dict(), 'active-models/vnet-model-csipo.mdl')
                torch.save(ventral_net.state_dict(), 'active-models/ventral-net-csipo.mdl')
                torch.save(dorsal_net.state_dict(), 'active-models/dorsal-net-csipo.mdl')
                torch.save(class_to_pos_net.state_dict(), 'active-models/posnet-model-csipo.mdl')
                torch.save(pos_to_class_net.state_dict(), 'active-models/colnet-model-csipo.mdl')
                torch.save(uv_to_class_net.state_dict(), 'active-models/uvtoclass-model-csipo.mdl')
                torch.save(cae.state_dict(), 'active-models/cae-model-csipo.mdl')


    plt.plot(episodes, losses)
    plt.show()


    
