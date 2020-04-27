import torch
import torch.nn as nn
from enum import Enum
import config

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