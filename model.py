import torch.nn as nn
import torch

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv1.weight)
        # Conv. 1 + ReLU + MaxPooling
        self.conv_block1 = nn.Sequential(conv1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # xavier initializer, prevent gradient exploding/diminishing
        torch.nn.init.xavier_uniform_(conv2.weight)
        
        # Conv. 2 + ReLU + MaxPooling
        self.conv_block2 = nn.Sequential(conv2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2))

        conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv3.weight)

        # Conv. 3 + ReLU + MaxPooling
        self.conv_block3 = nn.Sequential(conv3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(conv4.weight)

        # Conv. 4 + ReLU + MaxPooling
        self.conv_block4 = nn.Sequential(conv4,
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))
        
        # FC. 1 + ReLU + Droupout
        self.fc_block1 = nn.Sequential(nn.Linear(in_features=2048, out_features=1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.6))
        
        # FC. 2 + ReLU + Droupout
        self.fc_block2 = nn.Sequential(nn.Linear(in_features=1024, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.6))
        # FC. 3 + Softmax
        self.output = nn.Sequential(nn.Linear(in_features=256, out_features=10),
                                    nn.Softmax(dim=1))

    def forward(self, input):
        output = self.conv_block1(input)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)

        output = output.view(output.size()[0], -1)
        output = self.fc_block1(output)
        output = self.fc_block2(output)
        final_out = self.output(output)
        return final_out