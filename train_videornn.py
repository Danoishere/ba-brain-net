
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


def train_video_rnn(queue, lock, torchDevice, load_model=True):

    now = datetime.now()
    current_time = now.strftime("-%H-%M-%S")
    writer = SummaryWriter('tensorboard/train' + current_time, flush_secs=10)

    lr=config.lr
    batch_size = config.batch_size
    sequence_length = config.sequence_length
    w, h = config.w, config.h

    colors = config.colors
    shapes = config.shapes
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

    class_has_below_above_net = net.ClassHasObjectBelowAboveNet(torchDevice).to(torchDevice)
    class_has_below_above_net.load_state_dict(torch.load('active-models/classbelowabovenet-model.mdl', map_location=torchDevice))
    class_has_below_above_net.train()

    loss_aprox_net = net.LossApproximationNet(torchDevice).to(torchDevice)
    #loss_pred_net.load_state_dict(torch.load('active-models/loss-aprox-net-model.mdl', map_location=torchDevice))
    loss_aprox_net.train()

    q_net = net.QNet(torchDevice).to(torchDevice)
    #q_net.load_state_dict(torch.load('active-models/q-net-model.mdl', map_location=torchDevice))
    q_net.train()

    params = []
    params += list(cae.parameters())
    params += list(lgn_net.parameters())
    params += list(visual_cortex_net.parameters())
    params += list(class_to_pos_net.parameters())
    params += list(pos_to_class_net.parameters())
    params += list(uv_to_class_net.parameters())
    params += list(count_net.parameters())
    params += list(class_has_below_above_net.parameters())
    params += list(loss_aprox_net.parameters())
    params += list(q_net.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    #optimizer.load_state_dict(torch.load('active-models/optimizer.opt'))
    losses = []
    episodes = []

    episode = 0
    num_queries = config.num_queries
    skip = config.skip_factor

    
    while True:
        last_frame = 0
        batch_x, scenes = queue.get()
        # Pass batch frame by frame

        for repetition in range(3):
            frame = randint(0, sequence_length - 1)
            clip_length = 20 # randint(16, 24) + 1
            
            optimizer.zero_grad()
            lgn_net.init_hidden(torchDevice)
            
            last_action = 3
            last_v1_out = None
            clip_frame = 0

            memory = []
            
            for step in range(clip_length):
                clip_frame += 1

                action = choice(config.actions)
                frame += action

                current_frame = frame % sequence_length # int(frame*skip + offset)
                # print(current_frame)
                frame_input = torch.tensor(batch_x[current_frame], requires_grad=True).float().to(torchDevice)
                encoded = cae.encode(frame_input)
                output = encoded.reshape(batch_size, -1)
                output = lgn_net(output)
                last_frame = current_frame
                
                if clip_frame > 12:
                    v1_out = visual_cortex_net(output)

                    tot_loss_class_to_pos = []
                    tot_loss_pos_to_class = []
                    tot_loss_uv_to_class = []
                    tot_loss_countnet = []
                    tot_loss_class_has_below_above = []

                    # Sample 10 different objects combinations from each training batch.
                    for i in range(num_queries):
                        # Predict color, based on location
                        # y_target_pos = []
                        y_target_rel_pos = []
                        y_target_has_below_above = []
                        
                        all_uvs = []
                        obj_col_onehots = []
                        obj_col_indices = []
                        obj_shape_indices = []
                        obj_shape_onehots = []
                        below_above_indices = []
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
                            

                            obj_has_above = 'is_below' in scene_objects[rnd_obj].keys() #is below -> has above
                            obj_has_below = 'is_above' in scene_objects[rnd_obj].keys() #is above -> has below

                            if obj_has_below:
                                below_above_idx = belowAbove.index("below")

                            elif obj_has_above:
                                below_above_idx = belowAbove.index("above")
                            else:
                                below_above_idx = belowAbove.index("standalone")

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

                        # oh = one-hot
                        y_col_oh = torch.tensor(obj_col_onehots, requires_grad=True, dtype=torch.float32).to(torchDevice)
                        y_shape_oh = torch.tensor(obj_shape_onehots, requires_grad=True, dtype=torch.float32).to(torchDevice)
                        y_target_rel_pos_t = torch.tensor(y_target_rel_pos, dtype=torch.float32).to(torchDevice)
                        y_col_idx = torch.tensor(obj_col_indices, dtype=torch.long).to(torchDevice)
                        y_shape_idx = torch.tensor(obj_shape_indices, dtype=torch.long).to(torchDevice)
                        y_uvs = torch.tensor(all_uvs, requires_grad=True, dtype=torch.float32).to(torchDevice)
                        y_has_below_above_idx = torch.tensor(below_above_indices, dtype=torch.long).to(torchDevice)


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
                        y_pred_has_below_above = class_has_below_above_net(v1_out, y_col_oh, y_shape_oh)
                        tot_loss_class_has_below_above += [class_has_below_above_net.loss(y_pred_has_below_above,y_has_below_above_idx)]

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
                    tot_loss_countnet = count_net.loss(p_dones, p_cols, p_shapes, p_pos, scenes, last_frame)

                    tot_loss_class_to_pos = torch.stack(tot_loss_class_to_pos)
                    tot_loss_class_to_pos = torch.mean(tot_loss_class_to_pos,dim=0)

                    tot_loss_pos_to_class = torch.stack(tot_loss_pos_to_class)
                    tot_loss_pos_to_class = torch.mean(tot_loss_pos_to_class,dim=0)

                    tot_loss_uv_to_class = torch.stack(tot_loss_uv_to_class)
                    tot_loss_uv_to_class = torch.mean(tot_loss_uv_to_class,dim=0)

                    tot_loss_class_has_below_above = torch.stack(tot_loss_class_has_below_above)
                    tot_loss_class_has_below_above = torch.mean(tot_loss_class_has_below_above,dim=0)

                    print('Episode', episode,', Clip Frame',clip_frame,'Action', action, ', Loss Pos.:', torch.mean(tot_loss_class_to_pos).item())

                    tot_loss_sum = tot_loss_class_to_pos + tot_loss_pos_to_class + tot_loss_uv_to_class + tot_loss_countnet + tot_loss_class_has_below_above

                    # note to dano: 
                    # Problem: One loss for batch! That doesnt work (it's the fcking average). Use per-batch sums
                    # Also: Long term reward


                    #pred_norm_loss = loss_aprox_net(v1_out)
                    #tot_loss_loss_aprox = loss_aprox_net.loss(pred_norm_loss, norm_tot_loss)
                    #tot_loss_sum += tot_loss_loss_aprox

                    loss = torch.tensor(0.0, dtype=torch.float32).to(torchDevice)
                    if last_v1_out is not None:
                        current_loss = tot_loss_sum.clone().detach().float()
                        reward = (last_loss - current_loss).detach()
                        q_values = q_net(last_v1_out)
                        q_loss = q_net.loss(config.actions.index(last_action), q_values, reward)
                        loss += q_loss

                        writer.add_scalar("Loss/Q-Net-Loss", torch.mean(q_loss).item(), episode)
                        #memory.append((last_v1_out, last_action, reward))

                        #current_reward = torch.tensor(current_reward).float().to(torchDevice)
                        
                    last_loss = tot_loss_sum.clone().detach().float()
                    last_v1_out = v1_out

                    tot_loss_sum = torch.mean(tot_loss_sum)
                    
                    loss += tot_loss_sum
                    loss.backward(retain_graph=True)
                    #nn.utils.clip_grad_norm_(params, 0.025)
                    optimizer.step()
                    

                    episodes.append(episode)
                    writer.add_scalar("Loss/Class-to-Position-Loss", torch.mean(tot_loss_class_to_pos).item(), episode)
                    writer.add_scalar("Loss/Position-to-Class-Loss", torch.mean(tot_loss_pos_to_class).item(), episode)
                    writer.add_scalar("Loss/UV-to-Class-Loss", torch.mean(tot_loss_uv_to_class).item(), episode)
                    writer.add_scalar("Loss/Obj-Count-Loss", torch.mean(tot_loss_countnet).item(), episode)
                    writer.add_scalar("Loss/Class-has-Below-Above-Loss", torch.mean(tot_loss_class_has_below_above).item(), episode)
                   
                    #writer.add_scalar("Loss/Loss-Approximation-Loss", tot_loss_loss_aprox.item(), episode)
                    episode += 1

                    if episode % 500 == 0:
                        torch.save(lgn_net.state_dict(), 'active-models/lgn-net.mdl')
                        torch.save(visual_cortex_net.state_dict(), 'active-models/visual-cortex-net.mdl')
                        torch.save(class_to_pos_net.state_dict(), 'active-models/posnet-model.mdl')
                        torch.save(pos_to_class_net.state_dict(), 'active-models/colnet-model.mdl')
                        torch.save(uv_to_class_net.state_dict(), 'active-models/uvtoclass-model.mdl')
                        torch.save(cae.state_dict(), 'active-models/cae-model.mdl')
                        torch.save(count_net.state_dict(), 'active-models/countnet-model.mdl')
                        torch.save(class_has_below_above_net.state_dict(), 'active-models/classbelowabovenet-model.mdl')
                        torch.save(loss_aprox_net.state_dict(), 'active-models/loss-aprox-net-model.mdl')
                        torch.save(q_net.state_dict(), 'active-models/q-net-model.mdl')
                        torch.save(optimizer.state_dict(), 'active-models/optimizer.opt')

                last_action = action

            """
            rl_loss = torch.tensor(0.0, dtype=torch.float32).to(torchDevice)
            for i in range(len(memory)):
                mem = memory[i]
                v1_out = mem[0]
                action = mem[1]
                reward = mem[2].clone()

                q_values = q_net(v1_out)
                print(q_values)
                for r in range(i + 1, len(memory)):
                    future_mem = memory[i]
                    reward += 0.8**r * future_mem[2]

                q_loss = q_net.loss(config.actions.index(action), q_values, reward)
                rl_loss += q_loss*0.1

                writer.add_scalar("Loss/Q-Net-Loss", torch.mean(q_loss).item(), episode - len(memory) + i + 1)
            rl_loss.backward()
            """

    plt.plot(episodes, losses)
    plt.show()