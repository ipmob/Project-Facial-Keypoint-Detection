## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


## If device is have cuda installed use GPU else use CPU 
import torch.cuda

def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x



#Model


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        ## input 224*224
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        ## 224 ----> 224-5+1 = 220
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## 220/2 = 110 ( 32,110,110)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
        ## 110 ---> 110 - 3 + 1 = 108
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## 108 ---> 108/2 = 54 (64,54,54)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
        ## 54 ---> 54 -3 + 1 = 52
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## 52/2 = 26 (128,27,27)
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3)
        ## 26 - 3 +1 = 24
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## pooling reduce the size by 2 ( 24/2 = 12 )
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512 , kernel_size = 1)
        ## 12 -1+1 = 12
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ## 12/2 = 6

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features = 6 * 6 * 512, out_features = 1024) # The number of input gained by "print("Flatten size: ", x.shape)" in below
        self.fc2 = nn.Linear(in_features = 1024,    out_features = 512)
        self.fc3 = nn.Linear(in_features = 512,    out_features = 136) # the output 136 in order to having 2 for each of the 68 keypoint (x, y) pairs

        # Dropouts
        self.drop1 = nn.Dropout(p = 0.1)
        self.drop2 = nn.Dropout(p = 0.2)
        self.drop3 = nn.Dropout(p = 0.3)
        self.drop4 = nn.Dropout(p = 0.4)
        self.drop5 = nn.Dropout(p = 0.5)
        self.drop6 = nn.Dropout(p = 0.5)
        self.drop7 = nn.Dropout(p = 0.6)




    def forward(self, x):

        # First - Convolution + Activation + Pooling + Dropout
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.drop1(x)
#         print("First size: ", x.shape)

        # Second - Convolution + Activation + Pooling + Dropout
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
#         print("Second size: ", x.shape)

        # Third - Convolution + Activation + Pooling + Dropout
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
#         print("Third size: ", x.shape)

        # Forth - Convolution + Activation + Pooling + Dropout
        x = self.drop4(self.pool4(F.relu(self.conv4(x))))
        #print("Forth size: ", x.shape)
        x = self.drop5(self.pool5(F.relu(self.conv5(x))))
        #print("Fifth size: ", x.shape)

        # Flattening the layer
        x = x.view(x.size(0), -1)
#         print("Flatten size: ", x.shape)

        # First - Dense + Activation + Dropout
        x = self.drop6(F.relu(self.fc1(x)))
#         print("First dense size: ", x.shape)

        # Second - Dense + Activation + Dropout
        x = self.drop7(F.relu(self.fc2(x)))
        #print("Second dense size: ", x.shape)

        # Final Dense Layer
        x = self.fc3(x)
        #print("Final dense size: ", x.shape)

        return x
