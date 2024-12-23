import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Модель заточена под карту размера 6x6, 5 агентов и размер памяти 5
'''

# class GameNet(nn.Module):
#     '''
#     v0
#     '''
#     def __init__(self):
#         super(GameNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(30, 32, kernel_size=3, stride=1, padding=1),
#             nn.Tanh())
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.Tanh())
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.Tanh(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.fc1 = nn.Sequential(
#             nn.Linear(3*3*128, 256),
#             nn.Tanh())
#         self.fc2 = nn.Sequential(
#             nn.Linear(256, 36),
#             nn.LeakyReLU())
#
#
#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = x.reshape(-1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x

# class GameNet(nn.Module):
#     '''
#     v1
#     '''
#     def __init__(self):
#         super(GameNet, self).__init__()
#         self.fc1 = nn.Linear(in_features=30*6*6, out_features=1024)
#         self.fc2 = nn.Linear(in_features=1024, out_features=1024)
#         self.fc3 = nn.Linear(in_features=1024, out_features=36)
#         nn.init.uniform_(self.fc1.weight, -1, 1)
#         nn.init.uniform_(self.fc2.weight, -1, 1)
#         nn.init.uniform_(self.fc3.weight, -1, 1)
#
#     def forward(self, x):
#         x = torch.flatten(x)
#         x = nn.Tanh()(self.fc1(x))
#         x = nn.Tanh()(self.fc2(x))
#         x = nn.LeakyReLU()(self.fc3(x))
#         return x

# class GameNet(nn.Module):
#     '''
#     v2
#     '''
#     def __init__(self):
#         super(GameNet, self).__init__()
#         self.fc1 = nn.Linear(in_features=30*6*6, out_features=1024)
#         self.fc2 = nn.Linear(in_features=1024, out_features=1024)
#         self.fc3 = nn.Linear(in_features=1024, out_features=1024)
#         self.fc4 = nn.Linear(in_features=1024, out_features=1024)
#         self.fc5 = nn.Linear(in_features=1024, out_features=36)
#         nn.init.uniform_(self.fc1.weight, -1, 1)
#         nn.init.uniform_(self.fc2.weight, -1, 1)
#         nn.init.uniform_(self.fc3.weight, -1, 1)
#         nn.init.uniform_(self.fc4.weight, -1, 1)
#         nn.init.uniform_(self.fc5.weight, -1, 1)
#
#     def forward(self, x):
#         x = torch.flatten(x)
#         x = nn.Tanh()(self.fc1(x))
#         x = nn.Tanh()(self.fc2(x))
#         x = nn.Tanh()(self.fc3(x))
#         x = nn.Tanh()(self.fc4(x))
#         x = nn.LeakyReLU()(self.fc5(x))
#         return x

class GameNet(nn.Module):
    '''
    v3
    '''
    def __init__(self):
        super(GameNet, self).__init__()
        self.fc1 = nn.Linear(in_features=30*6*6, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=36)
        nn.init.uniform_(self.fc1.weight, -1, 1)
        nn.init.uniform_(self.fc2.weight, -1, 1)
        nn.init.uniform_(self.fc3.weight, -1, 1)
        nn.init.uniform_(self.fc4.weight, -1, 1)
        nn.init.uniform_(self.fc5.weight, -1, 1)

    def forward(self, x):
        x = torch.flatten(x)
        x = nn.LeakyReLU()(self.fc1(x))
        x = nn.LeakyReLU()(self.fc2(x))
        x = nn.LeakyReLU()(self.fc3(x))
        x = nn.LeakyReLU()(self.fc4(x))
        x = nn.LeakyReLU()(self.fc5(x))
        return x

if __name__ == '__main__':
    model = GameNet()
    test_input = torch.rand((1, 30, 6, 6))
    output = model(test_input)
    print(output)
    print(output.shape)