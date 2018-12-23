import torch.nn as nn


class net_basic(nn.Module):

    def __init__(self, top_bn=True):

        super(VAT, self).__init__()
        self.top_bn = top_bn
        self.main = nn.Sequential(
            # 3-D Convlution Neural Network

            # 64 * 64 * 64 * 1
            nn.Conv3d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            # 64 * 64 * 64 * 32
            nn.Conv3d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),

            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 32 * 32 * 32 * 32
            nn.Conv3d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            # 32 * 32 * 32 * 64
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),

            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 16 * 16 * 16 * 64
            nn.Conv3d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            # 16 * 16 * 16 * 128
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),

            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 8 * 8 * 18 * 128
            nn.Conv3d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),
            # 8 * 8 * 8 * 256
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(),

            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 4 * 4 * 4 * 256
            nn.Conv3d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),

            nn.Conv3d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(),

            nn.MaxPool3d(2, 2, 2),
            nn.Dropout3d(),

            # 2 * 2 * 2 * 512
            nn.Conv3d(512, 1024, 3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(),
            # 2 * 2 * 2 * 1024
            nn.Conv3d(1024, 1024, 3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(),

            nn.MaxPool3d((2, 2, 2)),
            nn.Dropout3d(),

            # 1 * 1 * 1 * 1024
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        # final flatten layer
        self.linear = nn.Linear(1024, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, input):
        output = self.main(input)
        output = self.linear(output.view(input.size()[0], -1))
        if self.top_bn:
            output = self.bn(output)
        return output
