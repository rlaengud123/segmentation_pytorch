import torch

from segmentation_models_pytorch import Unet, FPN

def fpn(backbone, pretrained_weights=None, classes = 1, activation = 'sigmoid'): 
    device = torch.device("cuda")
    model = FPN(encoder_name=backbone, 
                encoder_weights=pretrained_weights, 
                classes=classes, 
                activation=activation)
    model.to(device)
    model.eval() # 위치 확인해볼것
    
    return model

def unet(backbone, pretrained_weights=None, classes = 1, activation = 'sigmoid'): 
    device = torch.device("cuda")
    model = Unet(encoder_name=backbone, 
                 encoder_weights=pretrained_weights, 
                 classes=classes, 
                 activation=activation)
    model.to(device)
    model.eval() # 위치 확인해볼것
    
    return model