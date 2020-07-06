import torch.nn.functional as F
import torch.nn as nn
import torch

class encode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(encode_block, self).__init__()
        self.double = double
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        x = self.down_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x

class decode_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, double=False):
        super(decode_block, self).__init__()
        self.double = double
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=padding, stride = stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, True)
            )
        
    def forward(self, x):
        x = self.up_conv(x)
        if self.double:
            x = self.conv(x)
        
        return x

class WaveformEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1  = encode_block(in_channels=2, out_channels=64,kernel_size=(1,228),padding=(0,114),stride=(1,2))            
        self.enc2  = encode_block(64, 64, (1,128), (0,64), (1,3),False)
        self.enc3  = encode_block(64, 128, (1,64), (0,32), (1,3),False)    
        self.enc4  = encode_block(128, 256, (1,32), (0,16), (1,3),False)
        self.enc5  = encode_block(256, 512, (1,16), (0,8), (1,3),False)
        self.enc6  = encode_block(512, 512, (1,8), (0,4), (1,3),False)
        self.enc7  = encode_block(512, 512, (1,4), (0,2), (1,3),False)
        self.enc8  = encode_block(512, 1024, (1,3), (0,1), (1,3),False)
        # output 1024x1x1

    def forward(self, x1,x2):
        x0 = torch.cat([x1,x2],1)
        
        x = self.enc1(x0)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)

        return x

class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1  = encode_block(in_channels=6, out_channels=32,kernel_size=3,padding=1,stride=(2,1))            
        self.enc2  = encode_block(32, 64, 3, 1, (2,1),False)
        self.enc3  = encode_block(64, 128, 3, 1, (2,1),False)    
        self.enc4  = encode_block(128, 256, 3, 1, 2,False)
        self.enc5  = encode_block(256, 512, 3, 1, 2,False)
        self.enc6  = encode_block(512, 512, 3, 1, 2,False)
        self.enc7  = encode_block(512, 1024, 3, 1, 2,False)
        self.enc8  = encode_block(1024, 1024, (3,4), 1, 2,False)
        
        self.fc1   = fc(10*1024,1024)
        self.fc2   = fc(1024,1024)

    def forward(self, x1,x2):
        x0 = torch.cat([x1,x2],1)
        
        x = self.enc1(x0)
        
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        x = self.enc6(x)
        x = self.enc7(x)
        x = self.enc8(x)
        
        x = x.view(-1,10*1024)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

class DirectUpsampler(nn.Module):
    def __init__(self,output=128):
        super().__init__()

        # 4x4
        self.dec1  = decode_block(1024,512,4,0,1,True)
        # 8x8
        self.dec2  = decode_block(512,512,4,1,2,True)
        # 16x16
        self.dec3  = decode_block(512,256,4,1,2,True)
        # 32x32
        self.dec4  = decode_block(256,128,4,1,2,True)
        # 64x64
        self.dec5  = decode_block(128,128,4 if output >= 64 else 3,
                                    1,2 if output >= 64 else 1,True)
        # 128x128
        self.dec6  = decode_block(128,64,4 if output == 128 else 3,
                                    1, 2 if output == 128 else 1,True)

        self.final = nn.Conv2d(64, 1, 1) 

    def forward(self, x):
        if len(x) != 4:
            x = x.view(-1,1024,1,1)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.dec5(x)
        x = self.dec6(x)
        x = self.final(x)
        
        return x

class UNet(nn.Module):
    def __init__(self,output=128):
        super().__init__()

        self.enc1_0 = encode_block(1,64,3,1,1,False)
        self.enc1_1 = encode_block(64,64,3,1,1,False)
        self.epool1 = nn.MaxPool2d(2,2)
        
        self.enc2_0 = encode_block(64,128,3,1,1,False)
        self.epool2 = nn.MaxPool2d(2,2)
        
        self.enc3_0 = encode_block(128,256,3,1,1,False)
        self.epool3 = nn.MaxPool2d(2,2)
        
        self.enc4_0 = encode_block(256,256,3,1,1,False)
        self.epool4 = nn.MaxPool2d(2,2)
        
        self.bottleneck = encode_block(256,512,3,1,1,False)
        
        self.dec4_0 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dec4_1 = encode_block(512,256,3,1,1,False)
        
        self.dec3_0 = nn.ConvTranspose2d(256, 256, 4, 2, 1)
        self.dec3_1 = encode_block(512,256,3,1,1,False)
        
        self.dec2_0 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2_1 = encode_block(256,128,3,1,1,False)
        
        self.dec1_0 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1_1 = encode_block(128,128,3,1,1,False)

        self.dec0_0 = nn.ConvTranspose2d(128, 64, 4 if output >= 64 else 3, 2 if output >= 64 else 1, 1)
        self.dec0_1 = encode_block(64,64,3,1,1,False)

        self.dec00_0 = nn.ConvTranspose2d(64, 64, 4 if output == 128 else 3, 2 if output==128 else 1, 1)
        self.dec00_1 = encode_block(64,64,3,1,1,False)

        self.final  = nn.Conv2d(64,1,1)  
        # output 128x128x1

    def forward(self, x):        
        x = x.view(-1,1,32,32)

        x = self.enc1_0(x)
        x1 = self.enc1_1(x)
        x = self.epool1(x1)
        
        x2 = self.enc2_0(x)
        x = self.epool2(x2)
        
        x3 = self.enc3_0(x)
        x = self.epool3(x3)
        
        x4 = self.enc4_0(x)
        x = self.epool4(x4)
        
        x = self.bottleneck(x)
        
        x = torch.cat([self.dec4_0(x),x4],1)
        x = self.dec4_1(x)
        
        x = torch.cat([self.dec3_0(x),x3],1)
        x = self.dec3_1(x)
        
        x = torch.cat([self.dec2_0(x),x2],1)
        x = self.dec2_1(x)
        
        x = torch.cat([self.dec1_0(x),x1],1)
        x = self.dec1_1(x)

        x = self.dec0_0(x)
        x = self.dec0_1(x)

        x = self.dec00_0(x)
        x = self.dec00_1(x)
        
        x = self.final(x)
        
        return x
        
class fc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(fc, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_channels,out_channels),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        x = self.fc(x)
        
        return x

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)