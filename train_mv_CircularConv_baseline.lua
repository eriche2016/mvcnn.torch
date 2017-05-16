require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

require 'misc.optim_updates'
-- extra dependencies 
require 'misc.DataLoader'
local net_utils =  require 'misc.net_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Mutli-view CNN for 3D shape recognition')

cmd:text()
cmd:text('Options')

cmd:option('-silent', false, 'print opt to the screen?')
cmd:option('-seed', 1234, 'print opt to the screen?')
cmd:option('-save', './mv_logs', 'subdirectory to save logs') 
cmd:option('-batch_size', 16, 'batch size')

cmd:option('-optim','adam','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',4e-4,'learning rate')
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.999,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator for smoothing')

cmd:option('-backend', 'cudnn', 'whether we use cudnn backend, nn | cudnn')
cmd:option('-gpu_id', 1, 'GPU index')
cmd:option('-max_epoch', 200, 'maximum number of epochs')
cmd:option('-cnn_model', './models/pre_trained_net/imagenet_vgg_m_optnet.t7', 'path to cnn model that will be the base net for mv cnn:densenet-201.t7')
-- densenet:
-- pool_layer_idx = 50: mean class accuracy can achieve: 88.1  
-- pool_layer_idx = 75: mean class accuracy can achieve 90.75
-- pool_layer_idx = 80: mean class accuracy can achieve 89.6
-- pool_layer_idx = 100: mean class accuracy can achieve 87.7
-- imagenet_vgg_m_optnet(original paper): 
--  pool_layer_idx = 11: mean class accuracy can achieve: 90.125 

cmd:option('-pool_layer_idx', -1, 'pool out of the idx-th layer')
cmd:option('-input_h5', 'data/modelnet40.h5', 'h5 file that contains modelnet40 dataset ')

-- 
cmd:option('-init_from', '', 'specify checkpoint path to resume training.')

cmd:text()

-- print help or chosen options
opt = cmd:parse(arg)  
if not opt.silent then 
  print(opt)
end 

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpu_id >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then require 'cudnn' end
  cutorch.manualSeed(opt.seed)
  -- we have used CUDA_VISBILE_DEVICES = opt.gpu_id, so here we just set it on 1
  cutorch.setDevice(1) -- note +1 because lua is 1-indexed
end

-- create data loader 
loader = DataLoader(opt) 

print('Loading pretrained model...')
-- load features 

model = nil 
if false then 
    local model_raw = torch.load(opt.cnn_model):cuda()
    model_raw:remove(119) 
    model_raw:remove(118)
    model_raw:remove(117) 
    model_raw:remove(116) 

    model = model_raw -- bz x 1920 x 7 x 7
    -- add additional layer 
    model:add(nn.SpatialConvolution(1920, 40, 1, 1, 1, 1)) 
    model:add(nn.SpatialAveragePooling(7, 7)) 
    
    model:add(nn.Reshape(40)) 
    model:add(nn.LogSoftMax())
    print(model) 
elseif true then 
    local model_raw = torch.load(opt.cnn_model):cuda() 
    model_raw:remove(22)
    model_raw:remove(21) 
    
    model = model_raw
    -- add additional layer 
    model:add(nn.Linear(4096, 512))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(512, 40))

    model:add(nn.LogSoftMax()) 
    print(model)
    
else 
    model = nn.Sequential() 
    model:add(nn.SpatialConvolution(1, 96, 11, 11, 4, 4))
    model:add(nn.ReLU(true)) 
    model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2)) 
    model:add(nn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2)) 
    model:add(nn.ReLU(true)) 
    model:add(nn.SpatialCrossMapLRN(5, 0.0001, 0.75))
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- b x 256 x 12 x 12 

    model:add(nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)) 
    model:add(nn.ReLU(true)) 
    model:add(nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1)) 
    model:add(nn.ReLU(true)) 
    model:add(nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)) 
    model:add(nn.SpatialMaxPooling(3, 3, 2, 2))  -- b x 256 x 5 x 5 
    model:add(nn.ReLU(true)) 
    model:add(nn.SpatialConvolution(256, 40, 3, 3, 1, 1, 1, 1)) -- b x 40 x 5 x 5 
    model:add(nn.SpatialAveragePooling(5, 5)) -- b x 40 x 1 x 1 
    model:add(nn.Reshape(40)) 
    model:add(nn.LogSoftMax())
end 

if opt.gpu_id >= 0 then 
    model = model:cuda()
end 

-- set criterion
-- unused, criterion = dofile('torch_models/'..opt.model..'.lua')
-- assert(#model == #unused) -- check for consistency
if not criterion then
    criterion = nn.CrossEntropyCriterion():cuda()
end

-- construct pooling model from original one
if opt.pool_layer_idx < 1 then
    print('Select max pooling from which layer\'s output, type in layer index:')
    layer_idx = tonumber(io.read())
    print(layer_idx)
else
    layer_idx = opt.pool_layer_idx
end

model_pool = nil 
model_before_pool = nil 
if string.len(opt.init_from) > 0 then 
    print('loading model from checkpoint' .. opt.init_from)
    model_pool = torch.load(opt.init_from)
else 


    -- make pool model 
    model_copy_1 = model:clone()

    -- how many layers to remove: from bottom to up layers 
    -- extract features 
    for i = 1,layer_idx do
        model_copy_1:remove(1)
    end
    model_pool = model_copy_1 
    --[[
    model_pool = nn.Sequential() 
    model_pool:add(nn.Linear(2*512, 40))
    model_pool:add(nn.LogSoftMax())  
    --]] 

    -- make model before pool 
    model_copy_2 = model:clone()
    -- remove top layers 
    for i = layer_idx + 1, #model_copy_2 do 
        model_copy_2:remove(#model_copy_2)
    end 


    model_before_pool = model_copy_2
end 


-- this model is a model without learnable parameters 
--[[
merge_features_net = nn.Sequential() 
pt = nn.ParallelTable()   
-- branch 1: (bz * 12) x 512
branch1 = nn.Sequential() 
branch1:add(nn.View(-1, 12, 512)) 
-- bz x 512 
branch1:add(nn.Max(2)) 
-- bz x 512 
branch2 = nn.Sequential() 
branch2:add(nn.Identity())
pt:add(branch1)
pt:add(branch2)
merge_features_net:add(pt)
merge_features_net:add(nn.JoinTable(2))
--]] 


if opt.gpu_id > 0 then 
    model_pool = model_pool:cuda()
    model_before_pool = model_before_pool:cuda() 
    -- merge_features_net = merge_features_net:cuda()
end 

parameters, gradParameters = model_pool:getParameters()
print(model_pool)

parameters_0, gradParameters_0 = model_before_pool:getParameters() 
print(model_before_pool)

print('Loading data...')
--------------------------------------
-- compute mean and std for the dataset, which will be used in net_utils.prepro
--------------------------------------
local modelnet40_mean_tensor, modelnet40_std_tensor= loader:computeMeanStd()
modelnet40_mean = modelnet40_mean_tensor[1]
use_mean_only = true 
-- currenly, std is useless in net_utils.prepro
-- ## 
if use_mean_only then -- just reset it to 1  
    modelnet40_std = 1 
end 

---------------------------------------------------------------------------------------------
-- generate 12 masks 
--------------------------------------------------------------------------------------------
ind_row = torch.range(1, 12) 
mask_0 = torch.ByteTensor(12, 12):zero() 
function rotate_down(input, step)
    local output = input.new():resizeAs(input)
    local size = input:size(1)
    
    output[{{1, step}}] = input[{{size - step + 1, size}}]

    output[{{step+1, size}}] = input[{{1, size-step}}] 
    
    return output 
end 

 
for i = 1, ind_row:size(1) do 
    mask_0[i][i] = 1 
end

masks_dict = {} 
masks_dict[1] = mask_0

for k =1, 11 do 

    local mask_k = torch.ByteTensor(12, 12):zero()
    ind_col = rotate_down(ind_row, k) 
    for i = 1, ind_col:size(1) do 
        mask_k[i][ind_col[i]] = 1 
    end 

    masks_dict[k+1] = mask_k   
end 


---------------------------------------------------------------------------------------
-- Weights  
---------------------------------------------------------------------------------------
if opt.gpu_id >= 0 then 
    -- 12 views 
    Rotate_Kernels = torch.rand(512, 1, 12):cuda() 
end 

-- config logging
-- config logging
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% average instance accuracy (train set)', '% average instance accuracy (test set)',  '% average class accuracy (test set)'}
testLogger.showPlot = false

-- confusion matrix
confusion = optim.ConfusionMatrix(40)
confusion:zero()

----------------------------------------
-- Training routine
----------------------------------------
iter = 0 
epoch = nil
function train()
    model_before_pool:training()
    -- merge_features_net:training()
    model_pool:training() 

    epoch = epoch or 1 -- if epoch not defined, assign it as 1

    if epoch < 82 then  
        opt.learning_rate = 1e-3   
        -- according to original paper, we will divide the learning rate by 10 at 80 epochs and 120 epochs
    elseif epoch < 122 then 
        opt.learning_rate = 1e-4  -- learningRate = 0.01 afterwards
    else 
        opt.learning_rate = 1e-5   -- learningRate = 0.001 afterwards
    end 

    local tic = torch.tic()
    optim_state = {} 
    optim_state_0 = {} 
    optim_state_1 = {}

    while true  do 
        -- zero out grad 
        gradParameters:zero() 
        gradParameters_0:zero() 

        iter = iter + 1 
        -- load a batch of data 
        -- data.images_mv: bz x 12 x 1 x 224 x 224 
        local data = loader:getBatch(opt.batch_size, 'train')
        -- check whether we have done one epoch training 
        if data.wrapped then
            print('end one epoch, return')
            break;
        end 

        -- squeeze only first dimensiton 
        -- bz x 12 x 1 x 224 x 224 
        input = data.images_mv:squeeze(1)
        -- (bz*12) x 1 x 224 x 224 
        input = input:view(opt.batch_size*12, input:size(3), input:size(4), input:size(5))
        -- (bz * 12) x 3 x 224 x 224 
        input = input:expand(input:size(1), 3, input:size(3), input:size(4)) -- (opt.batch_size * 12) x 3 x 224 x 224
        
        -- preprocess input 
        input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augmen

        targets = data.label
        -- (bz*12) x 512 
        features = model_before_pool:forward(input) 

        -- bz x 512 x 12 x 1
        norm_view_module = nn.Sequential() 
        norm_view_module:add(nn.Normalize(1))
        -- norm_view_module:add(nn.View(-1, features:size(2), 12, 1)) 
        norm_view_module:add(nn.View(-1, 12, features:size(2))) 
        -- ##
        norm_view_module:add(nn.Max(2))
        -- ## 
        if opt.gpu_id >= 0 then 
            norm_view_module:cuda() 
        end 

        -- bz x 512 x 12 x 1 
        -- features_v = norm_view_module:forward(features)
        features_max_pool = norm_view_module:forward(features)

        --[[

        -- bz x 512 x 12 x 12 
        features_out_0 = features.new():resize(features_v:size(1), features_v:size(2), features_v:size(3), features_v:size(3))
        for z = 1, features_v:size(1) do 
            -- (512 x 12 x 1) * (512 x 1 x 12)
            features_out_0[z] = torch.bmm(features_v[z], Rotate_Kernels)
        end 

        -- bz x 512 x 12 
        features_out = features_v.new():resize(features_v:size(1), features_v:size(2), features_v:size(3))

        for b = 1, features_v:size(1) do
            -- expand mask: 512 x 12 x 12
            -- features_out_0[1]: 512 x 12 x 12 
            for k = 1, #masks_dict do 
                local mask_exp = masks_dict[k]:view(1, 12, 12):expandAs(features_out_0[1]):cuda() 
                -- 1 x 512 x 1
                features_out[{{b}, {}, {k}}] = torch.sum(torch.sum(torch.cmul(mask_exp, features_out_0[b]), 3), 2)
            end 
        end 
        
        -- bz x 512 
        max_module = nn.Max(3)
        if opt.gpu_id >= 0 then 
            max_module = max_module:cuda()
        end

        -- bz x 512 
        features_max_pool = max_module:forward(features_out)    
        --]] 

        -- ##
        -- merge 
        -- merged_features = merge_features_net:forward({features, features_max_pool}) 
        -- ## 

        -- forward to model_pool 
        outputs = model_pool:forward(features_max_pool) 
        -- ##
        -- outputs = model_pool:forward(merged_features)    
        -- ## 
        -- forward to criteiron 

        f = criterion:forward(outputs, targets)
        print('iter: ' .. iter .. ' , loss: ' .. f)

        -- backward to criterion 
        df_do = criterion:backward(outputs, targets) 

        dfeats_max_out = model_pool:backward(features_max_pool, df_do)
        -- ##
        -- dmerged_feats = model_pool:backward(merged_features, df_do)
        -- ## 
        -- ##
        -- dfeatures_0, dfeats_max_out = unpack(merge_features_net:backward({features, features_max_pool}, dmerged_feats))
        -- ## 
        -- bz x 512 x 12  
        -- dfeats_out = max_module:backward(features_out, dfeats_max_out)
        -- compute gradient 
        -- bz x 512 x 12 <- bz x 512 x 12 x 1, bz x 512 x 1 x 12  
        
        -- gradient w.r.t. input 
        -- rotate input  by 12 times 
        -- bz x 512 x 12 
        --[[ 
        dfeatures = features_v.new():resizeAs(features_v):zero()

        -- Rotate_Kernels: 512 x 1 x 12 
        -- 12 rotated kernels, which will be used to compute gradient 
        for b = 1, dfeatures:size(1) do 
            -- 512 X 12 
            for j = 1, 12 do
                if j == 1 then 
                    Rotate_Kernels_j = Rotate_Kernels:clone() 
                else 
                    Rotate_Kernels_j[{{}, {}, {j, Rotate_Kernels:size(3)}}] = Rotate_Kernels[{{}, {}, {1, Rotate_Kernels:size(3)-j+1}}]  
                    Rotate_Kernels_j[{{}, {}, {1, j-1}}] = Rotate_Kernels[{{}, {}, {Rotate_Kernels:size(3)-j+2, Rotate_Kernels:size(3)}}]
                end  

                --  512 x 12 = 512 x 1 x 12, bz x 512 x 12 
                dfeatures[{{b}, {}, {j}}] = torch.bmm(Rotate_Kernels, dfeats_out[b]:view(dfeats_out:size(2), dfeats_out:size(3), 1))
            end 
            
        end 

        -- gradient w.r.t. weight 
        -- 512 x 1 x 12 
        dRotate_Kernels = Rotate_Kernels.new():resizeAs(Rotate_Kernels):zero() 

        -- Rotate_Kernels: 512 x 1 x 12  
        for b = 1, dfeats_out:size(1) do 
            for j = 1, 12 do
                if j == 1 then 
                    -- bz x 512 x 12 
                    features_j = features_v:clone() 
                else 
                    features_j[{{}, {}, {j, features_v:size(3)}}] = features_v[{{}, {}, {1, features_v:size(3)-j+1}}]  
                    features_j[{{}, {}, {1, j-1}}] = features_v[{{}, {}, {features_v:size(3)-j+2, features_v:size(3)}}]
                end  
                -- 512 x 1 x 12, bz x 512 x 12 
                dRotate_Kernels[{{}, {}, {j}}] = dRotate_Kernels[{{}, {}, {j}}] + torch.bmm(dfeats_out[b]:view(dfeats_out:size(2), 1, dfeats_out:size(3)), features_j[b])
            end 
        end

        dRotate_Kernels:div(dfeats_out:size(1)*12) -- divide by batch size 

        -- backward to submodule 2 
        -- dfeatures: 
        dfeatures_raw = norm_view_module:backward(features, dfeatures:view(features_v:size(1), features_v:size(2), features_v:size(3), features_v:size(4)))
        -- ## 
        dfeatures_raw = dfeatures_raw -- + dfeatures_0
        -- ## 
        --
        model_before_pool:backward(input, dfeatures_raw)
        --]] 

        dfeatures_raw = norm_view_module:backward(features, dfeats_max_out)
        model_before_pool:backward(input, dfeatures_raw)

        -- clip gradient 
        gradParameters:clamp(-0.5, 0.5)
        gradParameters_0:clamp(-0.5, 0.5)
        -- dRotate_Kernels:clamp(-0.5, 0.5)

        -- parameter update
        if opt.optim == 'rmsprop' then
        rmsprop(parameter, gradParameters, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state)
        elseif opt.optim == 'adagrad' then
        adagrad(parameters, gradParameters, opt.learning_rate, opt.optim_epsilon, optim_state)
        elseif opt.optim == 'sgd' then
        sgd(parameters, gradParameters, opt.learning_rate)
        elseif opt.optim == 'sgdm' then
        sgdm(parameters, gradParameters, opt.learning_rate, opt.optim_alpha, optim_state)
        elseif opt.optim == 'sgdmom' then
        sgdmom(parameters, gradParameters, opt.learning_rate, opt.optim_alpha, optim_state)
        elseif opt.optim == 'adam' then
        adam(parameters, gradParameters, opt.learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state)
        else
        error('bad option opt.optim')
        end

        if opt.optim == 'rmsprop' then
        rmsprop(parameters_0, gradParameters_0, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state_0)
        elseif opt.optim == 'adagrad' then
        adagrad(parameters_0, gradParameters_0, opt.learning_rate, opt.optim_epsilon, optim_state_0)
        elseif opt.optim == 'sgd' then
        sgd(parameters_0, gradParameters_0, opt.learning_rate)
        elseif opt.optim == 'sgdm' then
        sgdm(parameters_0, gradParameters_0, opt.learning_rate, opt.optim_alpha, optim_state_0)
        elseif opt.optim == 'sgdmom' then
        sgdmom(parameters_0, gradParameters_0, opt.learning_rate, opt.optim_alpha, optim_state_0)
        elseif opt.optim == 'adam' then
        adam(parameters_0, gradParameters_0, opt.learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state_0)
        else
        error('bad option opt.optim')
        end
        --[[
        if opt.optim == 'rmsprop' then
        rmsprop(Rotate_Kernels, dRotate_Kernels, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, optim_state_1)
        elseif opt.optim == 'adagrad' then
        adagrad(Rotate_Kernels, dRotate_Kernels, opt.learning_rate, opt.optim_epsilon, optim_state_1)
        elseif opt.optim == 'sgd' then
        sgd(Rotate_Kernels, dRotate_Kernels, opt.learning_rate)
        elseif opt.optim == 'sgdm' then
        sgdm(Rotate_Kernels, dRotate_Kernels, opt.learning_rate, opt.optim_alpha, optim_state_1)
        elseif opt.optim == 'sgdmom' then
        sgdmom(Rotate_Kernels, dRotate_Kernels, opt.learning_rate, opt.optim_alpha, optim_state_1)
        elseif opt.optim == 'adam' then
        adam(Rotate_Kernels, dRotate_Kernels, opt.learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, optim_state_1)
        else
        error('bad option opt.optim')
        end
        --]] 

        if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
            confusion:batchAdd(outputs[#outputs], targets)
        else
            confusion:batchAdd(outputs, targets)    
        end
    end 

    confusion:updateValids()
    print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100

    confusion:zero()

    epoch = epoch + 1
end 


function test() 
    model_before_pool:evaluate()
    -- merge_features_net:evaluate()
    model_pool:evaluate() 

    local tic = torch.tic()
    local t = 1 
    while t <= loader.num_images_split['test'] do 
        -- load a batch of data 
        -- data.images_mv: bz x 12 x 1 x 224 x 224 
        local data = loader:getBatch(1, 'test')

        -- squeeze only first dimensiton 
        -- bz x 12 x 1 x 224 x 224 
        input = data.images_mv:squeeze(1)
        -- (bz*12) x 1 x 224 x 224 
        input = input:view(12, input:size(2), input:size(3), input:size(4))
        -- (bz * 12) x 3 x 224 x 224 
        input = input:expand(input:size(1), 3, input:size(3), input:size(4)) -- (opt.batch_size * 12) x 3 x 224 x 224
        
        -- preprocess input 
        input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augmen

        targets = data.label
        -- (bz*12) x 512 
        features = model_before_pool:forward(input) 

        -- bz x 512 x 12 x 1
        -- bz x 512 x 12 x 1
        norm_view_module = nn.Sequential() 
        norm_view_module:add(nn.Normalize(1))
        -- norm_view_module:add(nn.View(-1, features:size(2), 12, 1)) 
        norm_view_module:add(nn.View(-1, 12, features:size(2))) 
        -- ##
        norm_view_module:add(nn.Max(2))
        -- ## 
        if opt.gpu_id >= 0 then 
            norm_view_module:cuda() 
        end 
        -- bz x 512 x 12 x 1 
        features_v = norm_view_module:forward(features)
        --[[
        -- bz x 512 x 12 x 12 
        features_out_0 = features.new():resize(features_v:size(1), features_v:size(2), features_v:size(3), features_v:size(3))
        for z = 1, features_v:size(1) do 
            -- (512 x 12 x 1) * (512 x 1 x 12)
            features_out_0[z] = torch.bmm(features_v[z], Rotate_Kernels)
        end 

        -- bz x 512 x 12 
        features_out = features_v.new():resize(features_v:size(1), features_v:size(2), features_v:size(3))

        for b = 1, features_v:size(1) do
            -- expand mask: 512 x 12 x 12
            -- features_out_0[1]: 512 x 12 x 12 
            for k = 1, #masks_dict do 
                local mask_exp = masks_dict[k]:view(1, 12, 12):expandAs(features_out_0[1]):cuda() 
                -- 1 x 512 x 1
                features_out[{{b}, {}, {k}}] = torch.sum(torch.sum(torch.cmul(mask_exp, features_out_0[b]), 3), 2)
            end 
        end 
        
        -- bz x 512 
        max_module = nn.Max(3)
        if opt.gpu_id >= 0 then 
            max_module = max_module:cuda()
        end
        --]] 

        -- bz x 512 
        -- features_max_pool = max_module:forward(features_out)    
        -- features_max_pool = max_module:forward(features_out)   
        -- ##
        -- merge 
        -- merged_features = merge_features_net:forward({features, features_max_pool}) 
        -- ## 

        -- forward to model_pool 
        outputs = model_pool:forward(features_v) 
        -- ##
        -- outputs = model_pool:forward(merged_features)    
        -- ## 
        -- forward to criteiron 


        f = criterion:forward(outputs, targets)

        if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
            confusion:batchAdd(outputs[#outputs], targets)
        else
            confusion:batchAdd(outputs, targets)    
        end

        t = t  + 1 
    end 
    confusion:updateValids()
    print(('Test accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))

    test_acc = confusion.totalValid * 100

    confusion:zero()

end 

for i =1, opt.max_epoch do 
    test() 
    collectgarbage()  
    train()
    collectgarbage()  
end 
