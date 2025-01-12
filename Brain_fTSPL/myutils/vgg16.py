import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        feat_1_1 = F.relu(self.conv1_1(X))
        feat_1_2 = self.conv1_2(feat_1_1)
        relu_1_2 = F.relu(feat_1_2)
        feat_1 = F.max_pool2d(relu_1_2, kernel_size=2, stride=2)

        feat_2_1 = F.relu(self.conv2_1(feat_1))
        feat_2_2 = self.conv2_2(feat_2_1)
        relu_2_2 = F.relu(feat_2_2)
        feat_2 = F.max_pool2d(relu_2_2, kernel_size=2, stride=2)

        feat_3_1 = F.relu(self.conv3_1(feat_2))
        feat_3_2 = F.relu(self.conv3_2(feat_3_1))
        feat_3_3 = self.conv3_3(feat_3_2)
        relu_3_3 = F.relu(feat_3_3)
        feat_3 = F.max_pool2d(relu_3_3, kernel_size=2, stride=2)

        feat_4_1 = F.relu(self.conv4_1(feat_3))
        feat_4_2 = F.relu(self.conv4_2(feat_4_1))
        feat_4_3 = self.conv4_3(feat_4_2)
        relu_4_3 = F.relu(feat_4_3)

        return [relu_1_2, relu_2_2, relu_3_3, relu_4_3, feat_1_2, feat_2_2, feat_3_3, feat_4_3]
