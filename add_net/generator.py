import torch
from torch import nn
from torch.nn import functional as F


class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        
        self.bn=None
        if norm:
            self.bn = nn.BatchNorm2d(outplanes)
        
    def forward(self, x):
        fx = self.lrelu(x)
        fx = self.conv(fx)
        
        if self.bn is not None:
            fx = self.bn(fx)
            
        return fx

    
class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(outplanes)       
        
        self.dropout=None
        if dropout:
            self.dropout = nn.Dropout2d(p=0.5, inplace=True)
            
    def forward(self, x):
        fx = self.relu(x)
        fx = self.deconv(fx)
        fx = self.bn(fx)

        if self.dropout is not None:
            fx = self.dropout(fx)
            
        return fx

    
class Generator(nn.Module):
    """Encoder-Decoder model"""
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)
        
        self.decoder8 = DecoderBlock(512, 512, dropout=True)
        self.decoder7 = DecoderBlock(512, 512, dropout=True)
        self.decoder6 = DecoderBlock(512, 512, dropout=True)
        self.decoder5 = DecoderBlock(512, 512)
        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # encoder forward
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)
        # decoder forward
        d8 = self.decoder8(e8)
        d7 = self.decoder7(d8)
        d6 = self.decoder6(d7)
        d5 = self.decoder5(d6)
        d4 = self.decoder4(d5)
        d3 = self.decoder3(d4)
        d2 = F.relu(self.decoder2(d3))
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)
    
    
class UnetGenerator(nn.Module): # 适合512*512的数据
    """Unet-like Encoder-Decoder model"""
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) #512-256
        self.encoder2 = EncoderBlock(64, 128)  #256-128
        self.encoder3 = EncoderBlock(128, 256) #128-64
        self.encoder4 = EncoderBlock(256, 512) #64-32
        self.encoder5 = EncoderBlock(512, 512) #32-16
        self.encoder6 = EncoderBlock(512, 512) #16-8
        self.encoder7 = EncoderBlock(512, 512) #8-4
        self.encoder8 = EncoderBlock(512, 512, norm=False)#4-2
        self.encoder9 = EncoderBlock(512, 512, norm=False)#4-1
        
        self.decoder9 = DecoderBlock(512, 512) # [batch,512,1,1]=>[batch,512,2,2]
        self.decoder8 = DecoderBlock(2*512, 512) # [batch,1024,2,2]=>[batch,512,4,4]
        self.decoder7 = DecoderBlock(2*512, 512) # [batch,1024,4,4]=>[batch,512,8,8]
        self.decoder6 = DecoderBlock(2*512, 512) # [batch,1024,8,8]=>[batch,512,16,16]
        self.decoder5 = DecoderBlock(2*512, 512) # [batch,1024,16,16]=>[batch,512,32,32]
        self.decoder4 = DecoderBlock(2*512, 256) #[batch,1024,32,32]>[batch,256,64,64]
        self.decoder3 = DecoderBlock(2*256, 128) # [batch,512,64,64]=>[batch,128,128,128]
        self.decoder2 = DecoderBlock(2*128, 64) # [batch,256,128,128]=>[batch,64,256,256]
        self.decoder1 = nn.ConvTranspose2d(2*64, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # encoder forward
        e1 = self.encoder1(x) #256
        e2 = self.encoder2(e1) #128
        e3 = self.encoder3(e2) #64
        e4 = self.encoder4(e3) #32
        e5 = self.encoder5(e4) #16
        e6 = self.encoder6(e5) #8
        e7 = self.encoder7(e6) #4
        e8 = self.encoder8(e7) #2
        e9 = self.encoder9(e8) #1
        # decoder forward + skip connections
        d9 = self.decoder9(e9) #2
        d9 = torch.cat([d9, e8], dim=1) 
        d8 = self.decoder8(d9) #4
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8) #8
        d7 = torch.cat([d7, e6], dim=1) #8
        d6 = self.decoder6(d7) #16
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)

class Unet256Generator(nn.Module): # 适合256*256的数据
    """Unet-like Encoder-Decoder model"""
    def __init__(self,):
        super().__init__()
        
        self.encoder1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1) #512-256
        self.encoder2 = EncoderBlock(64, 128)  #256-128
        self.encoder3 = EncoderBlock(128, 256) #128-64
        self.encoder4 = EncoderBlock(256, 512) #64-32
        self.encoder5 = EncoderBlock(512, 512) #32-16
        self.encoder6 = EncoderBlock(512, 512) #16-8
        self.encoder7 = EncoderBlock(512, 512) #8-4
        self.encoder8 = EncoderBlock(512, 512, norm=False)#4-2

        self.decoder8 = DecoderBlock(512, 512) # [batch,1024,2,2]=>[batch,512,4,4]
        self.decoder7 = DecoderBlock(2*512, 512) # [batch,1024,4,4]=>[batch,512,8,8]
        self.decoder6 = DecoderBlock(2*512, 512) # [batch,1024,8,8]=>[batch,512,16,16]
        self.decoder5 = DecoderBlock(2*512, 512) # [batch,1024,16,16]=>[batch,512,32,32]
        self.decoder4 = DecoderBlock(2*512, 256) #[batch,1024,32,32]>[batch,256,64,64]
        self.decoder3 = DecoderBlock(2*256, 128) # [batch,512,64,64]=>[batch,128,128,128]
        self.decoder2 = DecoderBlock(2*128, 64) # [batch,256,128,128]=>[batch,64,256,256]
        self.decoder1 = nn.ConvTranspose2d(2*64, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # encoder forward
        e1 = self.encoder1(x) #256
        e2 = self.encoder2(e1) #128
        e3 = self.encoder3(e2) #64
        e4 = self.encoder4(e3) #32
        e5 = self.encoder5(e4) #16
        e6 = self.encoder6(e5) #8
        e7 = self.encoder7(e6) #4
        e8 = self.encoder8(e7) #2
        # decoder forward + skip connections
        d8 = self.decoder8(e8) #4
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.decoder7(d8) #8
        d7 = torch.cat([d7, e6], dim=1) #8
        d6 = self.decoder6(d7) #16
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.decoder1(d2)
        
        return torch.tanh(d1)