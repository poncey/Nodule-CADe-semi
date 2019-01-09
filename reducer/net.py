import torch.nn as nn
import torch.nn.functional as F
import torch


class NetBasic(nn.Module):

    def __init__(self, top_softmax=True):

        super(NetBasic, self).__init__()
        self.top_softmax = top_softmax

        self.top_pool_s = nn.MaxPool3d((1, 1, 1))
        self.top_pool_l = nn.MaxPool3d((2, 2, 2))
        self.main = nn.Sequential(
            # 3-D Convolution Neural Network

            # 32 * 32 * 32 * 1
            nn.Conv3d(1, 32, 5),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            # 28 * 28 * 28 * 32
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 14 * 14 * 14 * 32
            nn.Conv3d(32, 64, 5),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            # 10 * 10 * 10 * 64
            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 5 * 5 * 5 * 64
            nn.Conv3d(64, 128, 5),
            nn.BatchNorm3d(128),
            nn.LeakyReLU()

            # 1 * 1 * 1 * 128
        )
        # final flatten layer for each input
        self.fc = nn.Linear(256, 2)

    def forward(self, input_s, input_l):

        # input with 32 * 32 * 32 shape nodules
        x1 = self.top_pool_s(input_s)
        x1 = self.main(x1)

        # input with 64 * 64 * 64 shape samples
        x2 = self.top_pool_l(input_l)
        x2 = self.main(x2)

        # concat two paths
        x = torch.cat((x1.view(-1, 128), x2.view(-1, 128)), dim=1)
        x = self.fc(x)

        if self.top_softmax:
            x = F.softmax(x, dim=1)

        return x
