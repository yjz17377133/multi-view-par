import torch
from torch import nn
from torchvision import models
from module.convnext import convnext_base, convnext_small, convnext_large, convnext_tiny, convnext_xlarge

class Resnet50(nn.Module):
    
    def __init__(self):
        super(Resnet50,self).__init__()
        self.module = module = models.resnet50(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
            module.maxpool,
            module.layer1,
            module.layer2,
            module.layer3,
            module.layer4,
        )
        self.avgpool = module.avgpool
        
    def forward(self, input):
        net = self.features(input) #(batch,2048,7,7)
        net = self.avgpool(net) #(batch,2048,1,1)
        return net


class Resnet101(nn.Module):
    
    def __init__(self):
        super(Resnet101,self).__init__()
        self.module = module =  models.resnet101(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.bn1,
            module.relu,
            module.maxpool,
            module.layer1,
            module.layer2,
            module.layer3,
            module.layer4,
        )
        self.avgpool = module.avgpool
        
    def forward(self, inp):
        x = self.features(inp) #(batch,2048,7,7)
        x = self.avgpool(x) #(batch,2048,1,1)
        return x

class GoogleNet(nn.Module):
    def __init__(self):
        super(GoogleNet,self).__init__()
        self.module = module =  models.googlenet(pretrained=True)
        self.features = nn.Sequential(
            module.conv1,
            module.maxpool1,
            module.conv2,
            module.conv3,
            module.maxpool2,
            module.inception3a,
            module.inception3b,
            module.maxpool3,
            module.inception4a,
            module.inception4b,
            module.inception4c,
            module.inception4d,
            module.inception4e,
            module.maxpool4,
            module.inception5a,
            module.inception5b,
        )
        self.avgpool = module.avgpool
        
    def forward(self, inp):
        x = self.features(inp) #(batch,2048,7,7)
        print(x.shape)
        return x

class Dense121(nn.Module):
    
    def __init__(self):
        super(Dense121,self).__init__()
        module = models.densenet121(pretrained=True)
        self.features= module.features
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        print(net.shape)
        return net
    
class Dense161(nn.Module):
    
    def __init__(self):
        super(Dense161,self).__init__()
        module = models.densenet161(pretrained=True)
        self.features= module.features
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        print(net.shape)
        return net

class Dense201(nn.Module):
    
    def __init__(self):
        super(Dense201,self).__init__()
        module = models.densenet201(pretrained=True)
        self.features= module.features
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        print(net.shape)
        return net

class AlexNet(nn.Module):
    
    def __init__(self):
        super(AlexNet,self).__init__()
        module = models.alexnet(pretrained=True)
        self.features= module.features
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        print(net.shape)
        return net



class Vgg19(nn.Module):
    
    def __init__(self):
        super(Vgg19,self).__init__()
        module = models.vgg19(pretrained=True)
        self.features= module.features
        self.avgpool = module.avgpool
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        net = self.avgpool(net)   #(batch,512,7,7)
        return net
    


class Vgg11(nn.Module):
    
    def __init__(self):
        super(Vgg11,self).__init__()
        module = models.vgg11(pretrained=True)
        self.features= module.features
        self.avgpool = module.avgpool
    
    def forward(self, input):
        net = self.features(input)  #(batch,512,7,7)
        net = self.avgpool(net)   #(batch,512,7,7)
        return net


class Conv_Base(nn.Module):
    
    def __init__(self):
        super(Conv_Base,self).__init__()
        self.module = module = convnext_base(pretrained=True)
        #module = module.cuda()
        #self.features= module.forward_features
        self.features = nn.Sequential(
            module.downsample_layers[0],
            module.stages[0],
            module.downsample_layers[1],
            module.stages[1],
            module.downsample_layers[2],
            module.stages[2],
            module.downsample_layers[3],
            module.stages[3],
        )


    def forward(self, input):
        net = self.module(input)  #(batch,1024,7,7)
        print(net.shape)
        net2 = self.features(input)
        print(net2.shape)
        return net

class Conv_Large(nn.Module):
    
    def __init__(self):
        super(Conv_Large,self).__init__()
        self.module = module = convnext_large(pretrained=True)
        #module = module.cuda()
        #self.features= module.forward_features
        self.features = nn.Sequential(
            module.downsample_layers[0],
            module.stages[0],
            module.downsample_layers[1],
            module.stages[1],
            module.downsample_layers[2],
            module.stages[2],
            module.downsample_layers[3],
            module.stages[3],
        )
    
    def forward(self, input):
        net = self.module(input)  #(batch,1024,7,7)
        print(net.shape)
        net2 = self.features(input)
        print(net2.shape)
        return net

class Conv_Xlarge(nn.Module):
    
    def __init__(self):
        super(Conv_Xlarge,self).__init__()
        self.module = module = convnext_xlarge(pretrained=True)
        #module = module.cuda()
        #self.features= module.forward_features
        self.features = nn.Sequential(
            module.downsample_layers[0],
            module.stages[0],
            module.downsample_layers[1],
            module.stages[1],
            module.downsample_layers[2],
            module.stages[2],
            module.downsample_layers[3],
            module.stages[3],
        )
    
    def forward(self, input):
        net = self.module(input)  #(batch,1024,7,7)
        print(net.shape)
        net2 = self.features(input)
        print(net2.shape)
        return net

class Conv_Small(nn.Module):
    
    def __init__(self):
        super(Conv_Small,self).__init__()
        self.module = module = convnext_small(pretrained=True)
        #module = module.cuda()
        #self.features= module.forward_features
        self.features = nn.Sequential(
            module.downsample_layers[0],
            module.stages[0],
            module.downsample_layers[1],
            module.stages[1],
            module.downsample_layers[2],
            module.stages[2],
            module.downsample_layers[3],
            module.stages[3],
        )
    
    def forward(self, input):
        net = self.module(input)  #(batch,1024,7,7)
        print(net.shape)
        net2 = self.features(input)
        print(net2.shape)
        return net

class Conv_Tiny(nn.Module):
    
    def __init__(self):
        super(Conv_Tiny,self).__init__()
        self.module = module = convnext_tiny(pretrained=True)
        #module = module.cuda()
        #self.features= module.forward_features
        self.features = nn.Sequential(
            module.downsample_layers[0],
            module.stages[0],
            module.downsample_layers[1],
            module.stages[1],
            module.downsample_layers[2],
            module.stages[2],
            module.downsample_layers[3],
            module.stages[3],
        )
    
    def forward(self, input):
        net = self.module(input)  #(batch,1024,7,7)
        print(net.shape)
        net2 = self.features(input)
        print(net2.shape)
        return net




if __name__ =='__main__':
    #net = Conv_Base()
    net = Conv_Large()
    #net = GoogleNet()
    #net = Vgg11()
    #net = Vgg19()
    #net = AlexNet()
    #net = Dense161()
    #net = Dense121()
    #net = swin_L_224_22k()
    #net = EfficientNetB0()
    tmp_data = torch.randn(4,3,224,224)
    net = net.cuda()
    tmp_data=tmp_data.cuda()
    s = net(tmp_data)
    print('hello')

backbone_collection={'Alex':AlexNet, 'Resnet50': Resnet50, 'Resnet101':Resnet101, 'GoogleNet': GoogleNet, 'Dense121': Dense121, 'Dense161': Dense161, 'Dense201': Dense201, 'ConvBase':Conv_Base, 'ConvLarge':Conv_Large, 'ConvXlarge':Conv_Xlarge, 'ConvSmall':Conv_Small, 'ConvTiny':Conv_Tiny, 'Vgg11': Vgg11, 'Vgg19': Vgg19} 

#backbone_collection={'Resnet50': Resnet50, 'Resnet101':Resnet101, 'GoogleNet': GoogleNet, 'Dense121': Dense121, 'Dense161': Dense161, 'Dense201': Dense201, 'ConvBase':Conv_Base, 'ConvLarge':Conv_Large, 'ConvXlarge':Conv_Xlarge, 'ConvSmall':Conv_Small, 'ConvTiny':Conv_Tiny, 'Vgg11': Vgg11, 'Vgg19': Vgg19, 'tresnetl': tresnetl, 'tresnetl_v2': tresnetl_v2, 'tresnetxl': tresnetxl, 'CvT_w24_PETA': CvT_w24_PETA, 'CvT_w24_RAP': CvT_w24_RAP, 'swin_L_384_22k': swin_L_384_22k, 'swin_L_224_22k': swin_L_224_22k, 'swin_B_384_22k': swin_B_384_22k, 'swin_B_224_22k': swin_B_224_22k, 'Res2Net200_vd': Res2Net200_vd}    