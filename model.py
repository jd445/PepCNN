import torch
from torch import nn
class PepCNN(nn.Module):
    def __init__(self, gap_constraint=5, item_size=22):
        super(PepCNN, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64 * 24, (item_size, gap_constraint), 1, 0),
            nn.BatchNorm2d(64 * 24),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )


        self.fc = nn.Sequential(
            nn.Linear(24*64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        out = self.cnn(x)
        out, _ = out.max(-1)
        out = out.view(out.size()[0], -1)
        out = torch.squeeze(out)
        return self.fc(out)
