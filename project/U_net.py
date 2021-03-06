# import torch 
# import torch.nn as nn
# import torch.nn.functional as fn

# def double_conv(in_c, out_c):
#     conv = nn.Sequential(
#         nn.Conv2d(in_c, out_c, kernel_size=3),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_c, out_c, kernel_size=3),
#         nn.ReLU(inplace=True)
#     )

#     return conv

# def crop_img(tensor, target_tensor):
#     target_size = target_tensor.size()[2] #c, bs, w, h 
#     tensor_size = tensor.size()[2]
#     delta = tensor_size - target_size 
#     delta = delta // 2
#     return tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]



# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.down_conv1 = double_conv(3,64) #changed to 3 because RGB image
#         self.down_conv2 = double_conv(64,128)
#         self.down_conv3 = double_conv(128,256)
#         self.down_conv4 = double_conv(256,512)
#         self.down_conv5 = double_conv(512,1024)

#         self.up_trans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2 ,stride=2)

#         self.up_conv1 = double_conv(1024, 512)

#         self.up_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2 ,stride=2)

#         self.up_conv2 = double_conv(512, 256)

#         self.up_trans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2 ,stride=2)

#         self.up_conv3 = double_conv(256, 128)

#         self.up_trans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2 ,stride=2)

#         self.up_conv4 = double_conv(128, 64)

#         self.out = nn.Conv2d(64, 1, 1)



#     def forward(self, image):
#         #encoder
#         x1 = self.down_conv1(image) #
#         x2 = self.max_pool_2x2(x1)

#         x3 = self.down_conv2(x2) #
#         x4 = self.max_pool_2x2(x3)

#         x5 = self.down_conv3(x4) #
#         x6 = self.max_pool_2x2(x5)

#         x7 = self.down_conv4(x6) #
#         x8 = self.max_pool_2x2(x7)

#         x9 = self.down_conv5(x8)

#         #decoder
#         x = self.up_trans1(x9)
#         y = crop_img(x7, x)
#         x = self.up_conv1(torch.cat([x,y], 1))

#         x = self.up_trans2(x)
#         y = crop_img(x5, x)
#         x = self.up_conv2(torch.cat([x,y], 1))

#         x = self.up_trans3(x)
#         y = crop_img(x3, x)
#         x = self.up_conv3(torch.cat([x,y], 1))

#         x = self.up_trans4(x)
#         y = crop_img(x1, x)
#         x = self.up_conv4(torch.cat([x,y], 1))
        
#         x = self.out(x)

#         return x 

# if __name__ == "__main__":
#     network = UNet()
#     img = torch.rand((1,3,572,572))
#     print(network(img).size())

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()