require 'cudnn'
require 'nn'
require 'cunn'
opt = require 'optnet'

matio = require 'matio'

matconv_dir = './pre_trained_net/imagenet-matconvnet-vgg-m.mat'
save_torch_model_dir = './pre_trained_net/imagenet_vgg_m.t7'

local function convertMatConv2Torch(matconv_dir, save_dir) 

  print('start to load model .. ')
  matioModel = matio.load(matconv_dir)
  print('finishing loading model.')
  for i = 1, #matioModel.layers do
    local layer = matioModel.layers[i]
    local name = ''
    for k = 1, layer.name[1]:size(1) do
      name = name .. string.char(layer.name[1][k])
    end

    if string.find(name, 'conv') then
      print(i .. '-th layer ' .. name)
      print(name)
      print('weight: '); print(layer.weights[1]:size());print(layer.weights[2]:size()) 
      print('stride: '); print(layer.stride)
      print('pad: '); print(layer.pad)
    end
    -- assume using max pooling 
    if string.find(name, 'pool') then
      print(i .. '-th layer ' .. name)
      print('kernel: '); print(layer.pool)
      print('stride: '); print(layer.stride)
      print('pad: '); print(layer.pad)
    end 

    -- in mat conv net, implement fc layer using conv 
    if string.find(name, 'fc') then
      print(i .. '-th layer ' .. name)
      print(name)
      print('weight: '); print(layer.weights[1]:size());print(layer.weights[2]:size()) 
      print('stride: '); print(layer.stride)
      print('pad: '); print(layer.pad)
    end

    if string.find(name, 'relu') then 
      print(i .. '-th layer ' .. name)
    end
    -- last layer softmax layer  
    if string.find(name, 'prob') then 
      print(i .. '-th layer ' .. name)
    end 

  end

  model = nn.Sequential() 
  model:add(cudnn.SpatialConvolution(3,96,7,7,2,2,0,0))       -- 224 -> 55
  model:add(cudnn.ReLU(true))
  --model:add(nn.SpatialCrossMapLRN(3,0.00005,0.75))
  model:add(nn.SpatialMaxPooling(3,3,2,2,0, 0))                   -- 55 ->  27

  model:add(cudnn.SpatialConvolution(96,256,5,5,2,2,1,1))       -- 27 ->  27
  model:add(cudnn.ReLU(true))
  -- model:add(nn.SpatialCrossMapLRN(3,0.00005,0.75))
  -- pad: 0 1 0 1 
  model:add(nn.SpatialMaxPooling(3,3,2,2,1,1))                  -- 27 ->  13

  model:add(cudnn.SpatialConvolution(256,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(cudnn.SpatialConvolution(512,512,3,3,1,1,1,1))      -- 13 ->  13
  model:add(cudnn.ReLU(true))
  model:add(nn.SpatialMaxPooling(3, 3, 2,2, 0, 0))                   -- 13 -> 6
  model:add(cudnn.SpatialConvolution(512,4096,6,6,1,1,0,0))

  model:add(nn.View(4096))
  model:add(nn.Dropout(0.5))
  model:add(nn.ReLU(true))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(4096, 4096))
  model:add(nn.ReLU(true))
  model:add(nn.Linear(4096, 1000))
  model:add(nn.LogSoftMax())  

  Param_Matio_Model_idx = {1, 4, 7, 9, 11, 14, 16, 18}
  iidx =1 
  for i = 1, model:size() do
    local module_name = torch.typename(model:get(i))
    idx = Param_Matio_Model_idx[iidx]

    if module_name == 'cudnn.SpatialConvolution' or
      module_name == 'nn.SpatialConvolution' then
      local mat_weights = matioModel.layers[idx].weights[1]
      local mat_bias = matioModel.layers[idx].weights[2]
      local mat_weights = mat_weights:transpose(1,4):transpose(2,3)
      print(mat_weights:size());print(model:get(i).weight:size());
      model:get(i).weight:copy(mat_weights) -- copy weight
      model:get(i).bias:copy(mat_bias) 
      iidx = iidx + 1
    end

    if module_name == 'nn.Linear' then
      local mat_weights = matioModel.layers[idx].weights[1]
      local mat_weights = mat_weights:transpose(1,4):transpose(2,3)
      print(mat_weights:size());print(model:get(i).weight:size())

      local linear_weights = mat_weights:reshape(mat_weights:size(1), mat_weights:size(2)*mat_weights:size(3)*mat_weights:size(4)):transpose(1, 2)
      mat_bias =  matioModel.layers[idx].weights[2]
      model:get(i).weight:copy(linear_weights)
      model:get(i).bias:copy(mat_bias)
      iidx = iidx + 1;
    end
  end
    
  model:clearState() 
    
  -- optimizenet to redue memory
  model = model:cuda() 
  opts = {inplace=true, mode='training'}
  input = torch.rand(1, 3, 224, 224)
  optnet.optimizeMemory(model, input, opts)
     
  print('saved converted model to: ' .. save_dir)
  torch.save(save_dir, model)
end 

-- call function to convert matconv model to torch model 
convertMatConv2Torch(matconv_dir, save_torch_model_dir)
