require 'nn'
require 'cunn'
require 'cudnn'

-- currently support 12 views 
-- inputs will be Tensors of size: bz x 12 x C x H x W
encoder = nn.Sequential() 
encoder:add(nn.SpatialConvolution(1, 96, 11, 11, 4, 4))
encoder:add(nn.ReLU(true)) 
encoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
encoder:add(nn.SpatialMaxPooling(3, 3, 2, 2)) 


encoder:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2)) 
encoder:add(nn.ReLU(true)) 
encoder:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
encoder:add(nn.SpatialMaxPooling(3, 3, 2, 2)) 

encoder:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)) 
encoder:add(nn.ReLU(true)) 
encoder:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1)) 
encoder:add(nn.ReLU(true)) 
encoder:add(nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)) 
encoder:add(nn.ReLU(true)) 
encoder:add(nn.SpatialMaxPooling(3, 3, 2, 2)) 


mv_share_net = nn.ParallelTable()
-- siamese style 
mv_share_net:add(encoder) 

for k = 1, 11 do 
	mv_share_net:add(encoder:clone('weight','bias', 'gradWeight','gradBias'))
end 

model = nn.Sequential() 
-- first we split bz x 12 x C x H x W along 12 
model:add(nn.SplitTable(2)) -- {bz x C x H x W, bz x C x H x W, ..., bz x C x H x W} of 12 Tensors 
model:add(mv_share_net)  -- output will be a table 

model:add(nn.CMaxTable())

return model 

