import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

def linfinity(x, config): 
    d = torch.zeros_like(x).uniform_(-config.epsilon, config.epsilon)
    return torch.clamp(x+d, min=config.min, max=config.max)
def _linfinity(config): 
    return lambda x: linfinity(x[0],config)

def rotation(config): 
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomRotation(config.degree, fill=(0,)), 
        transforms.ToTensor()
        ])
    return torch.cat([t(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1)
def _rotation(config): 
    return lambda x: rotation(x[0],config)

def pair_rotation(x, config):
    np.random.seed(0)
    degree = (np.random.rand() - 0.5) * (config.degree * 2)
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomRotation(degrees=(degree - 1e-5, degree), fill=(0,)),
#         TF.rotate(degree), 
        transforms.ToTensor()
        ])
    t2 = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomRotation(degrees=(degree * 2. - 1e-5, degree * 2), fill=(0,)),
#         TF.rotate(degree * 2.), 
        transforms.ToTensor()
        ])
    part1 = torch.cat([t(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1) 
    part2 = torch.cat([t2(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1) 
    res = torch.cat([part1, part2], dim=0)
    return res
def _pair_rotation(config):
    return lambda x: pair_rotation(x[0], config)

def rts(x, config): 
    t = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.RandomAffine(config.angle, scale=config.scale, fillcolor=0), 
        transforms.RandomCrop(config.crop_sz,padding=config.padding),
        transforms.ToTensor()
        ])
    return torch.cat([t(x[i]) for i in range(x.size(0))], dim=0).unsqueeze(1)
def _rts(config): 
    return lambda x: rts(x[0],config)

hs = {
    "linfinity": _linfinity, 
    "rotation": _rotation, 
    "pair_rotation": _pair_rotation,
    "rts": _rts, 
    "dataloader": lambda config: (lambda x: x[2]),
    "none": lambda config: (lambda x: x[0])
}