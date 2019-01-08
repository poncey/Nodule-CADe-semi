import torch.nn as nn
import torch.nn.functional as F
import torch


class NetBasic(nn.Module):

    def __init__(self, top_bn=True):

        super(NetBasic, self).__init__()
        self.top_bn = top_bn

        self.top_pool_s = nn.MaxPool3d((1, 1, 1))
        self.top_pool_l = nn.MaxPool3d((2, 2, 2))
        self.main = nn.Sequential(
            # 3-D Convolution Neural Network

            # 32 * 32 * 32 * 1
            nn.Conv3d(1, 32, 5),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            # 28 * 28 * 28 * 32
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 14 * 14 * 14 * 32
            nn.Conv3d(32, 64, 5),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            # 10 * 10 * 10 * 64
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 5 * 5 * 5 * 64
            nn.Conv3d(64, 128, 5),
            nn.BatchNorm3d(128),
            nn.ReLU()

            # 1 * 1 * 1 * 128
        )
        # final flatten layer for each input
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, input_s, input_l):

        # input with 32 * 32 * 32 shape nodules
        x1 = self.top_pool_s(input_s)
        x1 = self.main(x1)

        # input with 64 * 64 * 64 shape samples
        x2 = self.top_pool_l(input_l)
        x2 = self.main(x2)

        # concat two paths
        x = torch.cat((x1.view(-1, 128), x2.view(-1, 128)), dim=1)
        x = self.fc1(F.relu(x))
        x = self.fc2(F.relu(x))

        if self.top_bn:
            x = self.bn(x)

        return x
