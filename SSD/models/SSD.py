import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
from layers.l2norm import *

def build_SSD(phase, num_classes, pretrained=True):
    base = vgg16()
    extras = add_extras(1024)
    loc, conf = boxbox(base, extras, num_classes)
    model = SSD(
            phase='train', 
            base = base,
            extras = extras,
            loc = loc,
            conf = conf,
            num_classes = 21,
            pretrained = pretrained
        )
    return model



class SSD(nn.Module):

    def __init__(self, phase, base, loc, conf, extras, num_classes, pretrained=True):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = 300

        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        self.l2norm = l2norm(512,20)

    def forward(self, x):
        """
        Args:
            x: input image or batch of images shape (batch,3,300,300)
        """
        sources = []
        loc = []
        conf = []

        for k in range(23):
            x = self.base[k](x)

        s = self.l2norm(x)
        sources.append(s)

        for k in range(23, 35):
            x = self.base[k](x)

        sources.append(x)

        for k, v in enumerate(self.extras):
            x = v(x)
            if k % 4 == 2:
                sources.append(x)

        for x, l, c in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == 'test':
            pass
        
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(loc.size(0), -1, self.num_classes)
            )

        return output





        




def vgg16(pretrained = True):
    vgg16 = models.vgg16(pretrained)

    layers_conv3_3 = (vgg16.features[i] for i in range(16))
    layer_pool3 = (nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
    layers_conv5_3 = (vgg16.features[i] for i in range(17,30))
    layer_pool5 = (nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
    layer_conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    layer_conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    layers = []
    layers += [*(layers_conv3_3), (layer_pool3), *(layers_conv5_3), 
        (layer_pool5), (layer_conv6), nn.ReLU(inplace=True), 
        (layer_conv7), nn.ReLU(inplace=True)]

    return layers

def add_extras(in_channel):
    cfg = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    layers = []
    in_c = in_channel
    flag = False
    Sflag = False

    for v in cfg:
        if v == 'S':
            Sflag = True
        else:
            if Sflag:
                layers += [nn.Conv2d(in_c, v, kernel_size=(1, 3)[flag], stride=2, padding=1), nn.ReLU(inplace=True)]
                Sflag=False
            else:
                layers += [nn.Conv2d(in_c, v, kernel_size=(1, 3)[flag]), nn.ReLU(inplace=True)]
            flag = not flag
            in_c = v

    return layers

def boxbox(vgg, extra_layer, num_classes):
    cfg =[4, 6, 6, 6, 4, 4] # number of boxes per feature map location

    loc_layers = []
    conf_layers = []
    vgg_source = [21, 33]

    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    for k, v in enumerate(extra_layer[2::4], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes, kernel_size=3, padding=1)]

    return (loc_layers, conf_layers)



if __name__ == '__main__':
    print("constructing model...")
    
    head = vgg16()
    extras = add_extras(1024)
    (loc, conf) = boxbox(head, extras, 20)

    model = SSD(head, extras, loc, conf)
    for name, layer in model._modules.items():
            print(name,layer)







