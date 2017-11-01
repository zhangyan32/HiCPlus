import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import gzip
import sys
import torch.optim as optim
conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5

class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        return x
'''
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
'''
'''
net = Net(40, 24)



#sys.exit()
#low_resolution_samples = low_resolution_samples.reshape((low_resolution_samples.shape[0], 40, 40))
#print low_resolution_samples[0:1, :,: ,: ].shape
#low_resolution_samples = torch.from_numpy(low_resolution_samples[0:1, :,: ,: ])
#X = Variable(low_resolution_samples)
#print X
#Y = Variable(torch.from_numpy(Y[0]))
#X = Variable(torch.randn(1, 1, 40, 40))
#print X
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
criterion = nn.MSELoss()
for epoch in range(2):  # loop over the dataset multiple times
    print "epoch", epoch

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        #print(inputs.size())
        #print(labels.size())
        #print type(inputs)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print outputs
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        print i
        # print statistics
        #print type(loss)
        #print loss
        #print loss.data[0]
        #print loss.data
        #print type(data), len(data)
        #print "the key is ", type(data[0])
        


print('Finished Training')


output = net(X)
print(output)
print type(output)

loss = criterion(output, Y)


net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.weight.grad)

'''

