require 'nn'
require 'cunn'
require 'cutorch'
require 'cudnn'

model = torch.load('./pre_trained_net/imagenet_vgg_m_optnet.t7') 

