import torch
import torch.nn as nn
from enum import Enum
import config
import numpy as np

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


# (vent_in, dors_in, shape, col) -> pos
class ClassToPosNet(nn.Module):
    def __init__(self):
        super(ClassToPosNet, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 256, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 64)
        self.fc6 = nn.Linear(64, 3)

        self.side_fc1 = nn.Linear(12, 64)
        self.side_fc2 = nn.Linear(64, 256)
        self.side_fc3 = nn.Linear(256, 256)


    def forward(self, v1_in, col, shape):

        side = torch.cat((col, shape), 1)
        side = self.side_fc1(side)
        side = self.lrelu(side)
        side = self.side_fc2(side)
        side = self.lrelu(side)
        side = self.side_fc3(side)
        side = self.lrelu(side)

        out = torch.cat((v1_in, side), 1)
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

        return out


    def loss(self, y_pred_pos, y_target_pos_t):
        # euclidean loss
        # sqrt(x^2 + y^2 + z^2)
        diff = torch.sum((y_pred_pos - y_target_pos_t)**2, dim=1)
        diff_sum_sqrt = torch.sqrt(diff)
        loss_pos = torch.mean(diff_sum_sqrt)
        return loss_pos

# (vent_in, dors_in, pos) -> (col, shape)
class PosToClass(nn.Module):
    def __init__(self, torchDevice):
        super(PosToClass, self).__init__()
        self.col_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.shape_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 64)

        self.logits_col = nn.Linear(64, 7)
        self.logits_shape = nn.Linear(64, 5)

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
        out = self.lrelu(out)
        out = self.fc6(out)
        out = self.lrelu(out)
        
        col = self.logits_col(out)
        shape = self.logits_shape(out)

        return col, shape

    def loss(self, pred_col_logits, pred_shape_logits, target_col_idx, target_shape_idx):
        col_loss = self.col_criterion(pred_col_logits, target_col_idx)
        shape_loss = self.shape_criterion(pred_shape_logits, target_shape_idx)
        total_loss = col_loss + shape_loss
        return total_loss

# (vent_in, dors_in, uv) -> (col, shape)
class UVToClass(nn.Module):
    def __init__(self, torchDevice):
        super(UVToClass, self).__init__()
        self.col_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.shape_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 64)

        self.logits_col = nn.Linear(64, 7)
        self.logits_shape = nn.Linear(64, 5)


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
        
        col = self.logits_col(out)
        shape = self.logits_shape(out)

        return col, shape

    def loss(self, pred_col_logits, pred_shape_logits, target_col_idx, target_shape_idx):
        col_loss = self.col_criterion(pred_col_logits, target_col_idx)
        shape_loss = self.shape_criterion(pred_shape_logits, target_shape_idx)
        total_loss = col_loss + shape_loss
        return total_loss




class ObjCountNet(nn.Module):
    def __init__(self, torchDevice):
        super(ObjCountNet, self).__init__()
        self.col_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.shape_criterion = nn.CrossEntropyLoss().to(torchDevice)
        self.done_criterion = nn.BCEWithLogitsLoss().to(torchDevice)

        self.lrelu = nn.LeakyReLU()
        self.n_layers = 1
        self.hidden_dim = 1024

        self.fc1 = nn.Linear(2048, 2048)
        self.rnn = nn.LSTM(2048, self.hidden_dim, self.n_layers)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)

        self.binary_done = nn.Linear(256, 1)
        self.logits_col = nn.Linear(256, 7)
        self.logits_shape = nn.Linear(256, 5)
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
        l_col = []
        l_shape = []
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
            col = self.logits_col(obj_out)
            shape = self.logits_shape(obj_out)
            pos = self.obj_pos(obj_out)

            l_done.append(done)
            l_col.append(col)
            l_shape.append(shape)
            l_pos.append(pos)

        return l_done, l_col, l_shape, l_pos

    def dist_loss(self, y_pred_pos, y_target_pos_t):
        # euclidean loss
        # sqrt(x^2 + y^2 + z^2)
        diff = torch.sum((y_pred_pos - y_target_pos_t)**2)
        diff_sum_sqrt = torch.sqrt(diff)
        loss_pos = torch.mean(diff_sum_sqrt)
        return loss_pos

    def loss(self, l_has_more, l_col, l_shape, l_pos, scenes, last_frame):
        scene_objects = []
        for scene in scenes:
            objs = []
            for obj in scene['objects']:
                objs.append((obj, scene['objects'][obj]))
            scene_objects.append(objs)

        loss_pos = [] #torch.tensor(0.0).to(self.torchDevice)
        loss_col = []
        loss_shape = []
        # 1 = found more, 0 = all objects outputted
        loss_has_more = []
        for o in range(10):
            for s in range(len(scenes)):
                scene = scenes[s]
                scene_objs = scene_objects[s]

                if len(scene_objs) > 0:
                    scene_pos_pred = l_pos[o][s].clone().detach().cpu().numpy()
                    scene_cam_mat = np.array(scene["cam_base_matricies"][last_frame])
                    scene_cam_mat = np.linalg.inv(scene_cam_mat)
                
                    closest_dist = 100
                    closest_obj = None

                    for scene_obj in scene_objs:
                        scene_obj_pos = scene_cam_mat @ np.array(scene_obj[1]['pos'] + [1.0])
                        scene_obj_pos = scene_obj_pos[:3]
                        dist = np.linalg.norm(scene_obj_pos - scene_pos_pred)

                        if dist < closest_dist:
                            closest_dist = dist
                            closest_obj = scene_obj

                    obj_pos = torch.tensor(closest_obj[1]['pos']).to(self.torchDevice)
                    obj_col_idx = torch.tensor([config.colors.index(closest_obj[1]['color-name'])], dtype=torch.long).to(self.torchDevice)
                    obj_shape_idx = torch.tensor([config.shapes.index(closest_obj[0].split('-')[0])], dtype=torch.long).to(self.torchDevice)
                    obj_has_more = torch.tensor([1.0], dtype=torch.float).to(self.torchDevice)

                    loss_pos += [self.dist_loss(l_pos[o][s], obj_pos)]
                    loss_col += [self.col_criterion(l_col[o][s].unsqueeze(0), obj_col_idx)]
                    loss_shape += [self.shape_criterion(l_shape[o][s].unsqueeze(0), obj_shape_idx)]
                    loss_has_more += [self.done_criterion(l_has_more[o][s], obj_has_more)]
                    scene_objects[s].remove(closest_obj)

                else:
                    obj_has_more = torch.tensor([0.0], dtype=torch.float).to(self.torchDevice)
                    loss_has_more += [self.done_criterion(l_has_more[o][s], obj_has_more)]


        loss_pos = torch.mean(torch.stack(loss_pos))
        loss_col = torch.mean(torch.stack(loss_col))
        loss_shape = torch.mean(torch.stack(loss_shape))
        loss_has_more = torch.mean(torch.stack(loss_has_more))
                
        return loss_pos + loss_col + loss_shape + loss_has_more

# (vent_in, dors_in, shape, col) -> hasAboveNet
class ClassHasObjectBelowAboveNet(nn.Module):
    def __init__(self, torchDevice):
        super(ClassHasObjectBelowAboveNet, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 256, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 64)
        self.fc5 = nn.Linear(64, 3)

        self.side_fc1 = nn.Linear(12, 64)
        self.side_fc2 = nn.Linear(64, 256)
        self.side_fc3 = nn.Linear(256, 256)

        self.belowAbove_criterion = nn.CrossEntropyLoss().to(torchDevice)


    def forward(self, v1_in, col, shape):
        side = torch.cat(( col, shape), 1)
        side = self.side_fc1(side)
        side = self.lrelu(side)
        side = self.side_fc2(side)
        side = self.lrelu(side)
        side = self.side_fc3(side)
        side = self.lrelu(side)

        out = torch.cat((v1_in, side), 1)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.lrelu(out)
        out = self.fc3(out)
        out = self.lrelu(out)
        out = self.fc4(out)
        out = self.lrelu(out)
        out = self.fc5(out)
        return out


    def loss(self, y_pred_has_below_above, y_target_has_below_above_t):
        return self.belowAbove_criterion(y_pred_has_below_above, y_target_has_below_above_t)
