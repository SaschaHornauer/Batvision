from Utils import *

class TestNet(nn.Module):
    def __init__(self,generator,output=128):
        super().__init__()

        self.audio_encoder1 = WaveformEncoder()
        self.audio_encoder2 = SpectrogramEncoder()

        if generator == "direct":
            self.decoder = DirectUpsampler(output)
        elif generator == "unet":
            self.decoder = UNet(output)
        else:
            raise Exception("Generator: use direct or unet")

        self.fc1 = fc(2048,1024)
        self.fc2 = fc(1024,1024)

    def forward(self, x1,x2,x3,x4):   
        x_a1 = self.audio_encoder1(x3,x4)
        x_a2 = self.audio_encoder2(x1,x2)
        x_a1 = x_a1.view(-1,1024)
        x = torch.cat([x_a1,x_a2],1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1,1024,1,1)
        x = self.decoder(x)

        return x

class WaveformNet(nn.Module):
    def __init__(self,generator,output=128):
        super().__init__()

        self.audio_encoder = WaveformEncoder()
        if generator == "direct":
            self.decoder = DirectUpsampler(output)
        elif generator == "unet":
            self.decoder = UNet(output)
        else:
            raise Exception("Generator: use direct or unet")

    def forward(self, x1,x2):
        
        x = self.audio_encoder(x1,x2)
        x = self.decoder(x)
        
        return x

class SpectrogramNet(nn.Module):
    def __init__(self,generator,output=128):
        super().__init__()

        self.audio_encoder = SpectrogramEncoder()   
        if generator == "direct":
            self.decoder = DirectUpsampler(output)
        elif generator == "unet":
            self.decoder = UNet(output)
        else:
            raise Exception("Generator: use direct or unet")   
                        
        
    def forward(self, x1,x2):
        x = self.audio_encoder(x1,x2)
        x = self.decoder(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self,output=128):
        super().__init__()
        self.output = output

        if output == 128:
            final = 256
            k = 4
            s = [2,2,2]
        elif output == 64:
            final = 128
            k = 4
            s = [1,1,2]
        elif output == 32:
            final = 128
            k = 3
            s = [1,2,1]

        self.conv1 = nn.Conv2d(1, 64, k, s[0], 1)
        self.act1  = nn.LeakyReLU(0.2, True)

        self.conv2 = nn.Conv2d(64, 128, k, s[1], 1)
        self.bn2   = nn.BatchNorm2d(128)
        self.act2  = nn.LeakyReLU(0.2, True)

        if output == 128:
            self.conv3 = nn.Conv2d(128, 256, k, s[2], 1)
            self.bn3   = nn.BatchNorm2d(256)
            self.act3  = nn.LeakyReLU(0.2, True)

        self.final = nn.Conv2d(final,1,k,s[2],1)

        
    def forward(self,x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.output == 128:
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.act3(x)
        x = self.final(x)

        return x
