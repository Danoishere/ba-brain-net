import torch
import torch.nn as nn
from enum import Enum

class Query(Enum):
    POS = 1
    COL = 2

batch_size = 32

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_dim = 2048
        self.n_layers = 1
        self.rnn = nn.LSTM(2048, self.hidden_dim, self.n_layers)



    def init_hidden(self):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        self.hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda(),
                       torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda())


    def forward(self, frame):
        # (Sequence Length, Batch size, Inputs)
        out = frame.reshape(1, batch_size, -1)
        out, hidden = self.rnn(out, self.hidden)
        self.hidden = hidden
        out = out.reshape(batch_size, -1)
        return out


















# 5 shapes, 7 colors
# 200 + 12
class PosNet(nn.Module):
    def __init__(self):
        super(PosNet, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(2048 + 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 64)
        self.fc7 = nn.Linear(64, 3)

        self.fc_i1 = nn.Linear(12, 24)
        self.fc_i2 = nn.Linear(24, 64)
        self.fc_i3 = nn.Linear(64, 256)

    def forward(self, out,  shape, col):

        inp = torch.cat((shape, col), 1)
        inp = self.fc_i1(inp)
        inp = self.lrelu(inp)
        inp = self.fc_i2(inp)
        inp = self.lrelu(inp)
        inp = self.fc_i3(inp)
        inp = self.lrelu(inp)

        out = torch.cat((out, inp), 1)
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
        out = self.fc7(out)
        return out

# Pos in col out
class ColNet(nn.Module):
    def __init__(self):
        super(ColNet, self).__init__()
        self.fc1 = nn.Linear(2048 + 3, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 64)
        self.fc5 = nn.Linear(64, 10)
        self.fc6 = nn.Linear(10, 7)

    # give position, receive color
    def forward(self, state, pos):
        out = torch.cat((state, pos), 1)
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = torch.relu(out)
        out = self.fc3(out)
        out = torch.relu(out)
        out = self.fc4(out)
        out = torch.relu(out)
        out = self.fc5(out)
        out = torch.relu(out)
        out = self.fc6(out)
        return out

class GreenNet(nn.Module):
    def __init__(self):
        super(GreenNet, self).__init__()
        self.fc1 = nn.Linear(100, 20)
        self.fc2 = nn.Linear(20, 1)

    # give position, receive color
    def forward(self, out):
        out = self.fc1(out)
        out = torch.relu(out)
        out = self.fc2(out)
        out = out.reshape(batch_size)
        return out