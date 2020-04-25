
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
    cae.load_state_dict(torch.load('active-models/cae-model.mdl'))
    cae.train()

    lgn_net = net.VisionNet().cuda()
    lgn_net.load_state_dict(torch.load('active-models/lgn-net.mdl'))
    lgn_net.train()

    visual_cortex_net = net.VisualCortexNet().cuda()
    visual_cortex_net.load_state_dict(torch.load('active-models/visual-cortex-net.mdl'))
    visual_cortex_net.train()

    class_to_pos_net = net.ClassToPosNet().cuda()
    class_to_pos_net.load_state_dict(torch.load('active-models/posnet-model.mdl'))
    class_to_pos_net.train()

    """
    pos_to_class_net = net.PosToClass().cuda()
    #pos_to_class_net.load_state_dict(torch.load('active-models/colnet-model-csipo.mdl'))
    pos_to_class_net.train()

    uv_to_class_net = net.UVToClass().cuda()
    #uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model-csipo.mdl'))
    uv_to_class_net.train()
    """

    params = []
    params += list(cae.parameters())
    params += list(lgn_net.parameters())
    params += list(visual_cortex_net.parameters())
    params += list(class_to_pos_net.parameters())
    # params += list(pos_to_class_net.parameters())
    # params += list(uv_to_class_net.parameters())
    
    optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer.load_state_dict(torch.load('active-models/optimizer.opt'))
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
        frames = list(range(sequence_length))
        offset = randint(0, skip - 1)
        shuffle(frames)

        start_frame = randint(0, sequence_length - 1)
        clip_length = randint(8,16) + 1
        
        optimizer.zero_grad()
        lgn_net.init_hidden()

        clip_frame = 0
        for frame in range(start_frame, start_frame + clip_length):
            clip_frame += 1
            current_frame = frame % sequence_length # int(frame*skip + offset)
            print(current_frame)
            frame_input = torch.tensor(batch_x[current_frame], requires_grad=True).float().cuda()
            encoded = cae.encode(frame_input)
            output = encoded.reshape(batch_size, -1)
            output = lgn_net(output)
            last_frame = current_frame

            if clip_frame > 4:

                v1_out = visual_cortex_net(output)

                tot_loss_class_to_pos = [] #torch.tensor(0.0).cuda()
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
                    y_pred_pos = class_to_pos_net(v1_out, y_col_oh, y_shape_oh)
                    tot_loss_class_to_pos += [class_to_pos_net.loss(y_pred_pos, y_target_rel_pos_t)]

                    # Find class loss
                    """
                    y_pred_col, y_pred_shape = pos_to_class_net(v1_out, y_target_rel_pos_t)
                    tot_loss_pos_to_class += pos_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)

                    # UV to class loss
                    y_pred_col, y_pred_shape = uv_to_class_net(v1_out, y_uvs)
                    tot_loss_uv_to_class += uv_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)

                    """

                tot_loss_stack = torch.stack(tot_loss_class_to_pos)
                tot_loss_class_to_pos = torch.mean(tot_loss_stack)

                print('Episode', episode, ', Loss Pos.:', tot_loss_class_to_pos.item())

                tot_loss_class_to_pos.backward(retain_graph=True)
                # nn.utils.clip_grad_norm_(params, 0.05)
                optimizer.step()
                
                episodes.append(episode)
                #losses.append(loss.item()/num_queries)

                writer.add_scalar("Loss/Class-to-Position-Loss", tot_loss_class_to_pos.item(), episode)
                episode += 1

                if episode % 500 == 0:
                    torch.save(lgn_net.state_dict(), 'active-models/lgn-net.mdl')
                    torch.save(visual_cortex_net.state_dict(), 'active-models/visual-cortex-net.mdl')
                    torch.save(class_to_pos_net.state_dict(), 'active-models/posnet-model.mdl')
                    torch.save(cae.state_dict(), 'active-models/cae-model.mdl')
                    torch.save(optimizer.state_dict(), 'active-models/optimizer.opt')


    plt.plot(episodes, losses)
    plt.show()


    
