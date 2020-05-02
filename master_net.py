

import net
from autoencoder import ConvAutoencoder

import torch
import torch.nn as nn
import config
from random import shuffle, randint
import numpy as np

class MasterNet:
    def __init__(self, torchDevice):
        self.torchDevice = torchDevice
        self.cae = ConvAutoencoder().to(torchDevice)
        self.cae.load_state_dict(torch.load('active-models/cae-model.mdl', map_location=torchDevice))
        self.cae.train()

        self.lgn_net = net.VisionNet().to(torchDevice)
        self.lgn_net.load_state_dict(torch.load('active-models/lgn-net.mdl', map_location=torchDevice))
        self.lgn_net.train()

        self.visual_cortex_net = net.VisualCortexNet().to(torchDevice)
        self.visual_cortex_net.load_state_dict(torch.load('active-models/visual-cortex-net.mdl', map_location=torchDevice))
        self.visual_cortex_net.train()

        self.class_to_pos_net = net.ClassToPosNet().to(torchDevice)
        self.class_to_pos_net.load_state_dict(torch.load('active-models/posnet-model.mdl', map_location=torchDevice))
        self.class_to_pos_net.train()

        self.pos_to_class_net = net.PosToClass(torchDevice).to(torchDevice)
        self.pos_to_class_net.load_state_dict(torch.load('active-models/colnet-model.mdl', map_location=torchDevice))
        self.pos_to_class_net.train()

        self.uv_to_class_net = net.UVToClass(torchDevice).to(torchDevice)
        self.uv_to_class_net.load_state_dict(torch.load('active-models/uvtoclass-model.mdl', map_location=torchDevice))
        self.uv_to_class_net.train()

        self.count_net = net.ObjCountNet(torchDevice).to(torchDevice)
        self.count_net.load_state_dict(torch.load('active-models/countnet-model.mdl', map_location=torchDevice))
        self.count_net.train()

        self.class_has_below_above_net = net.ClassHasObjectBelowAboveNet(torchDevice).to(torchDevice)
        #self.class_has_below_above_net.load_state_dict(torch.load('active-models/classbelowabovenet-model.mdl', map_location=torchDevice))
        self.class_has_below_above_net.train()

        self.params = []
        self.params += list(self.cae.parameters())
        self.params += list(self.lgn_net.parameters())
        self.params += list(self.visual_cortex_net.parameters())
        self.params += list(self.class_to_pos_net.parameters())
        self.params += list(self.pos_to_class_net.parameters())
        self.params += list(self.uv_to_class_net.parameters())
        self.params += list(self.count_net.parameters())
        self.params += list(self.class_has_below_above_net.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=config.lr)
        self.episode = 0


    def train_on_batch(self, batch_x, scenes, on_gradient_update = None):
        print("-------------------------------")

        # Random start frame
        start_frame = randint(0, config.sequence_length - 1)
        # Random clip length
        clip_length = randint(4, 8) + 1
        
        self.optimizer.zero_grad()
        self.lgn_net.init_hidden(self.torchDevice)
        self.on_gradient_update = on_gradient_update

        self.clip_frame = 0
        for frame in range(start_frame, start_frame + clip_length):
            frame = frame % config.sequence_length

            self.clip_frame += 1
            self.train_on_frame(frame, scenes, batch_x)

    def train_on_frame(self, frame, scenes, batch_x):
        print(frame)
        frame_input = torch.tensor(batch_x[frame], requires_grad=True).float().to(self.torchDevice)

        # CNN encoder
        output = self.cae.encode(frame_input)
        output = self.lgn_net(output)

        if self.clip_frame > 4:
            v1_out = self.visual_cortex_net(output)

            tot_loss_class_to_pos = []
            tot_loss_pos_to_class = []
            tot_loss_uv_to_class = []
            tot_loss_countnet = []
            tot_loss_class_has_below_above = []

            # Sample 10 different objects combinations from each training batch.
            for i in range(config.num_queries):
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
                below_above_onehots = []
                all_objs = []
                all_cam_pos = []

                for scene in scenes:
                    scene_objects = scene["objects"]
                    rnd_obj = np.random.choice(list(scene_objects.keys()))

                    last_frame_transf_mat = np.array(scene["cam_base_matricies"][frame])
                    last_frame_transf_mat_inv = np.linalg.inv(last_frame_transf_mat)

                    last_frame_uv = scene["ss_objs"][frame][rnd_obj]
                    all_uvs.append([last_frame_uv["screen_x"], last_frame_uv["screen_y"]])
                    
                    obj_pos = scene_objects[rnd_obj]['pos']
                    obj_rel_pos = last_frame_transf_mat_inv @ np.array(obj_pos + [1])
                    obj_rel_pos = obj_rel_pos[:3]

                    obj_col_idx = config.colors.index(scene_objects[rnd_obj]['color-name'])
                    obj_shape_idx = config.shapes.index(rnd_obj.split("-")[0])
                    

                    obj_has_above = 'is_below' in scene_objects[rnd_obj].keys() #is below -> has above
                    obj_has_below = 'is_above' in scene_objects[rnd_obj].keys() #is above -> has below

                    below_above_oh = np.zeros(len(config.belowAbove))

                    if obj_has_below:
                        below_above_idx = config.belowAbove.index("below")
                        below_above_oh[below_above_idx] = 1.0

                    elif obj_has_above:
                        below_above_idx = config.belowAbove.index("above")
                        below_above_oh[below_above_idx] = 1.0
                    
                    else:
                        below_above_idx = config.belowAbove.index("standalone")
                        below_above_oh[below_above_idx] = 1.0


                    obj_col_oh = np.zeros(len(config.colors))
                    obj_col_oh[obj_col_idx] = 1.0
                    obj_shape_oh = np.zeros(len(config.shapes))
                    obj_shape_oh[obj_shape_idx] = 1.0

                    #y_target_pos.append(obj_pos)
                    y_target_rel_pos.append(obj_rel_pos)
                    obj_col_onehots.append(obj_col_oh)
                    obj_shape_onehots.append(obj_shape_oh)
                    below_above_onehots.append(below_above_oh)
                    obj_col_indices.append(obj_col_idx)
                    obj_shape_indices.append(obj_shape_idx)
                    below_above_indices.append(below_above_idx)


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
                y_col_oh = torch.tensor(obj_col_onehots, requires_grad=True, dtype=torch.float32).to(self.torchDevice)
                y_shape_oh = torch.tensor(obj_shape_onehots, requires_grad=True, dtype=torch.float32).to(self.torchDevice)
                y_target_rel_pos_t = torch.tensor(y_target_rel_pos, dtype=torch.float32).to(self.torchDevice)
                y_col_idx = torch.tensor(obj_col_indices, dtype=torch.long).to(self.torchDevice)
                y_shape_idx = torch.tensor(obj_shape_indices, dtype=torch.long).to(self.torchDevice)
                y_uvs = torch.tensor(all_uvs, requires_grad=True, dtype=torch.float32).to(self.torchDevice)
                y_has_below_above_oh = torch.tensor(below_above_onehots, requires_grad=True, dtype=torch.float32).to(self.torchDevice)
                y_has_below_above_idx = torch.tensor(below_above_indices, dtype=torch.long).to(self.torchDevice)

                # Find position loss
                y_pred_pos = self.class_to_pos_net(v1_out, y_col_oh, y_shape_oh)
                tot_loss_class_to_pos += [self.class_to_pos_net.loss(y_pred_pos, y_target_rel_pos_t)]

                # Find class loss
                y_pred_col, y_pred_shape = self.pos_to_class_net(v1_out, y_target_rel_pos_t)
                tot_loss_pos_to_class += [self.pos_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)]

                # UV to class loss
                y_pred_col, y_pred_shape = self.uv_to_class_net(v1_out, y_uvs)
                tot_loss_uv_to_class += [self.uv_to_class_net.loss(y_pred_col,y_pred_shape, y_col_idx, y_shape_idx)]

                # Find hasAbove loss
                y_pred_has_below_above = self.class_has_below_above_net(v1_out, y_has_below_above_oh, y_col_oh, y_shape_oh)
                tot_loss_class_has_below_above += [self.class_has_below_above_net.loss(y_pred_has_below_above,y_has_below_above_idx)]

            p_dones, p_cols, p_shapes, p_pos = self.count_net(v1_out)
            tot_loss_countnet = self.count_net.loss(p_dones, p_cols, p_shapes, p_pos, scenes, frame)

            tot_loss_class_to_pos = torch.stack(tot_loss_class_to_pos)
            tot_loss_class_to_pos = torch.mean(tot_loss_class_to_pos)

            tot_loss_pos_to_class = torch.stack(tot_loss_pos_to_class)
            tot_loss_pos_to_class = torch.mean(tot_loss_pos_to_class)

            tot_loss_uv_to_class = torch.stack(tot_loss_uv_to_class)
            tot_loss_uv_to_class = torch.mean(tot_loss_uv_to_class)

            tot_loss_class_has_below_above = torch.stack(tot_loss_class_has_below_above)
            tot_loss_class_has_below_above = torch.mean(tot_loss_class_has_below_above)

            self.episode += 1
            if self.on_gradient_update is not None:
                self.on_gradient_update([tot_loss_class_to_pos, tot_loss_pos_to_class, tot_loss_uv_to_class, tot_loss_countnet, tot_loss_class_has_below_above])

    def perform_update(self, loss_list, retain_graph=True):
        tot_loss = torch.sum(torch.stack(loss_list))
        tot_loss.backward(retain_graph=retain_graph)
        #nn.utils.clip_grad_norm_(params, 0.025)

        print("Episode", self.episode, ", Total Loss:", tot_loss.item())
        self.optimizer.step()

        return tot_loss.item()