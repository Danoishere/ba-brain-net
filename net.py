import torch
import torch.nn as nn
from enum import Enum
import config
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class Query(Enum):
    POS = 1
    COL = 2

batch_size = config.batch_size

class VisionNet(nn.Module):
    def __init__(self):
        super(VisionNet, self).__init__()
        self.hidden_dim = 2048
        self.lrelu = nn.LeakyReLU()
        self.n_layers = 1
        self.rnn = nn.LSTM(2048, self.hidden_dim, self.n_layers)


    def init_hidden(self, torchDevice):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torchDevice),
                       torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torchDevice))


    def forward(self, frame):
        # (Sequence Length, Batch size, Inputs)
        out = frame.reshape(1, batch_size, -1)
        out, hidden = self.rnn(out, self.hidden)
        self.hidden = hidden
        out = out.reshape(batch_size, -1)
        out = self.lrelu(out)
        return out

class VisualCortexNet(nn.Module):
    def __init__(self):
        super(VisualCortexNet, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 2048)
        self.batchnorm2 = nn.BatchNorm1d(2048)

    def forward(self, v1_out):
        out = self.batchnorm1(v1_out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        out = self.lrelu(out)
        out = self.fc4(out)
        out = self.batchnorm2(out)
        return out


# (vent_in, dors_in, uv) -> (col, shape)
class UVToClass(nn.Module):
    def __init__(self, torchDevice):
        super(UVToClass, self).__init__()
        self.ripe_criterion = nn.BCEWithLogitsLoss(reduction='none').to(torchDevice)
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 64)

        self.binary_ripe = nn.Linear(64, 1)


    # give position, receive color
    def forward(self, v1_in, uv):
        out = torch.cat((v1_in, uv), 1)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        out = self.lrelu(out)
        out = self.fc4(out)
        out = self.lrelu(out)
        out = self.fc5(out)
        out = self.lrelu(out)
        out = self.fc6(out)
        out = self.lrelu(out)
        
        ripe = self.binary_ripe(out)
        return ripe.squeeze(1)

    def loss(self, pred_ripe_logits, target_ripe_idx):
        ripe_loss = self.ripe_criterion(pred_ripe_logits, target_ripe_idx)
        return ripe_loss



class ObjCountNet(nn.Module):
    def __init__(self, torchDevice):
        super(ObjCountNet, self).__init__()
        self.ripe_criterion = nn.BCEWithLogitsLoss(reduction='none').to(torchDevice)
        self.done_criterion = nn.BCEWithLogitsLoss(reduction='none').to(torchDevice)

        self.lrelu = nn.LeakyReLU()
        self.n_layers = 1
        self.hidden_dim = 1024

        self.fc1 = nn.Linear(2048, 2048)
        self.rnn = nn.LSTM(2048, self.hidden_dim, self.n_layers)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

        self.binary_done = nn.Linear(256, 1)
        self.binary_ripe = nn.Linear(256, 1)
        self.obj_pos = nn.Linear(256, 3)
        self.torchDevice = torchDevice

        

    def init_hidden(self, torchDevice):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torchDevice),
                       torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(torchDevice))


    def forward(self, v1_in):
        self.init_hidden(self.torchDevice)

        out = self.fc1(v1_in)
        out = self.lrelu(out)
        
        stop = False

        l_done = []
        l_ripe = []
        l_pos = []

        out = out.reshape(1,batch_size, -1)
        for i in range(10):
            obj_out, self.hidden = self.rnn(out, self.hidden)
            obj_out = obj_out.reshape(batch_size, -1)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc2(obj_out)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc3(obj_out)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc4(obj_out)
            obj_out = self.lrelu(obj_out)

            done = self.binary_done(obj_out)
            ripe = self.binary_ripe(obj_out)
            pos = self.obj_pos(obj_out)

            l_done.append(done)
            l_ripe.append(ripe)
            l_pos.append(pos)

        return l_done, l_ripe, l_pos


    def infere(self, v1_in):
        self.init_hidden(self.torchDevice)

        out = self.fc1(v1_in)
        out = self.lrelu(out)
        
        stop = False

        l_done = []
        l_ripe = []
        l_pos = []

        out = out.reshape(1,batch_size, -1)
        done_bool = False
        i = 0
        while not done_bool and i < 10:
            i += 1
            obj_out, self.hidden = self.rnn(out, self.hidden)
            obj_out = obj_out.reshape(batch_size, -1)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc2(obj_out)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc3(obj_out)
            obj_out = self.lrelu(obj_out)
            obj_out = self.fc4(obj_out)
            obj_out = self.lrelu(obj_out)

            done = torch.sigmoid(self.binary_done(obj_out))
            ripe =  torch.sigmoid(self.binary_ripe(obj_out))
            pos = self.obj_pos(obj_out)

            done_bool = done.squeeze().item() < 0.5

            if not done_bool:
                l_done.append(done)
                l_ripe.append(ripe)
                l_pos.append(pos)

        objs = []
        for tpl in zip(l_done, l_ripe, l_pos):
            key = (tpl[1].item() > 0.5, tpl[2].detach().numpy())
            objs.append(key)
            #if key not in objs:
            #    objs[key] = tpl

        return objs


    def dist_loss(self, y_pred_pos, y_target_pos_t):
        # euclidean loss
        # sqrt(x^2 + y^2 + z^2)
        diff = torch.sum((y_pred_pos - y_target_pos_t)**2)
        diff_sum_sqrt = torch.sqrt(diff)
        return diff_sum_sqrt

    def loss(self, l_has_more, l_ripe, l_pos, scenes, current_frame):
        scene_objects = []
        for scene in scenes:
            objs = []
            for obj in scene['objects']:
                objs.append((obj, scene['objects'][obj]))
            scene_objects.append(objs)

        pred_pos = torch.stack(l_pos)
        loss_batch = []
        for s in range(len(scenes)):
            loss_pos = [] #torch.tensor(0.0).to(self.torchDevice)
            loss_ripe = []
            # 1 = found more, 0 = all objects outputted
            loss_has_more = []
            # Calculate transformation matrix for relative positions
            scene_cam_mat = np.array(scene["cam_base_matricies"][current_frame[s]])
            scene_cam_mat = np.linalg.inv(scene_cam_mat)

            scene = scenes[s]
            scene_objs = scene_objects[s]
            scene_pos = []
            for scene_obj in scene_objs:
                scene_obj_pos = scene_cam_mat @ np.array(scene_obj[1]['pos'] + [1.0])
                scene_pos.append(scene_obj_pos[:3])
            scene_pos = np.asarray(scene_pos)

            pred_scene_pos_t = pred_pos[:len(scene_pos), s, :]
            pred_scene_pos_n = pred_scene_pos_t.clone().cpu().detach().numpy()

            cost_matrix = cdist(pred_scene_pos_n, scene_pos)
            _, assignment = linear_sum_assignment(cost_matrix)

            # REMEMBER: Use updated relative position!!!!!!!

            for i in range(len(scene_objs)):
                closest_obj = scene_objs[assignment[i]]
                closest_pos = scene_pos[assignment[i]]

                obj_pos = torch.tensor(closest_pos).to(self.torchDevice)
                ripe_val = 0.0
                if "ripe" in closest_obj[0]:
                    ripe_val = 1.0
                
                obj_ripe = torch.tensor([ripe_val], dtype=torch.float).to(self.torchDevice)
                obj_has_more = torch.tensor([1.0], dtype=torch.float).to(self.torchDevice)

                loss_pos += [self.dist_loss(l_pos[i][s], obj_pos)]
                loss_ripe += [self.ripe_criterion(l_ripe[i][s], obj_ripe)]
                loss_has_more += [self.done_criterion(l_has_more[i][s], obj_has_more)]
      
            for i in range(len(scene_objs), len(l_has_more)):
                obj_has_more = torch.tensor([0.0], dtype=torch.float).to(self.torchDevice)
                loss_has_more += [self.done_criterion(l_has_more[i][s], obj_has_more)]

            loss_pos = torch.mean(torch.stack(loss_pos))
            loss_ripe =  torch.mean(torch.stack(loss_ripe))
            loss_has_more =  torch.mean(torch.stack(loss_has_more))

            loss_batch.append(loss_pos + loss_ripe  + loss_has_more)
                
        return torch.stack(loss_batch)


class QNet(nn.Module):
    def __init__(self, torchDevice):
        super(QNet, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 64)
        self.fc4 = nn.Linear(64, 7)

        self.reward_criterion = nn.MSELoss().to(torchDevice)


    def forward(self, v1_in):
        out = self.fc1(v1_in)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        out = self.lrelu(out)
        out = self.fc4(out)
        return out


    def loss(self, selected_action_idx, y_pred_reward, y_target_reward):
        y_pred_reward_vec = torch.zeros_like(y_target_reward)
        for scene_idx in range(len(selected_action_idx)):
            y_pred_reward_vec[scene_idx] = y_pred_reward[scene_idx, selected_action_idx[scene_idx]]

        return self.reward_criterion(y_pred_reward_vec, y_target_reward)
        

#Input: Position
#Output: is_Ripe
class PosIsRipeClass(nn.Module):
    def __init__(self, torchDevice):
        super(PosIsRipeClass, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 1)

        self.ripe_criterion = nn.BCEWithLogitsLoss().to(torchDevice)

    def forward(self, v1_in, pos):
        out = torch.cat((v1_in, pos), 1)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        out = self.lrelu(out)
        out = self.fc4(out)
        out = self.lrelu(out)
        out = self.fc5(out)

        return out.squeeze(1)


    def loss(self, pred_isRipe, target_isRipe):
        ripe_loss = self.ripe_criterion(pred_isRipe, target_isRipe)
        return ripe_loss