import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, 3, 1, bias=False)
        self.fc0 = nn.Linear(28 * 28, 28 * 28, bias=False)
        self.fc1 = nn.Linear(28 * 28, 10, bias=False)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        x = F.relu(x)
        out_fc0 = x
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output, out_fc0

