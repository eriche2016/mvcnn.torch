require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'

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
cmd:option('-learning_rate', 1e-3, 'leanring rate')
cmd:option('-learning_rate_decay', 1e-7, 'learning_rate_decay')
cmd:option('-weight_decay', 5e-4, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-epoch_step', 20, 'epoch step, we decrease the learning rate by half')

cmd:option('-backend', 'cudnn', 'whether we use cudnn backend, nn | cudnn')
cmd:option('-gpu_id', 1, 'GPU index')
cmd:option('-max_epoch', 200, 'maximum number of epochs')
cmd:option('-cnn_model', './models/pre_trained_net/imagenet_vgg_m_optnet.t7', 'path to cnn model that will be the base net for mv cnn:./models/pre_trained_net/densenet-201.t7')
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
    model:add(nn.Linear(4096, 40)) 
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

local model_pool = nil 
local model_copy = nil 
if string.len(opt.init_from) > 0 then 
    print('loading model from checkpoint' .. opt.init_from)
    model_pool = torch.load(opt.init_from)
else 
    -- make pool model 
    model_copy = model:clone()

    -- how many layers to remove: from bottom to up layers 
    -- extract features 

    for i = 1,layer_idx do
        model_copy:remove(1)
    end
    model_pool = model_copy 
end 

if opt.gpu_id > 0 then 
    model_pool = model_pool:cuda()
end 

parameters, gradParameters = model_pool:getParameters()
print(model_pool)

print('Loading data...')
--------------------------------------
-- compute mean and std for the dataset, which will be used in net_utils.prepro
--------------------------------------
local modelnet40_mean_tensor, modelnet40_std_tensor= loader:computeMeanStd()
modelnet40_mean = modelnet40_mean_tensor[1]
use_mean_only = true 
-- currenly, std is useless in net_utils.prepro
if use_mean_only then -- just reset it to 1  
    modelnet40_std = 1 
end 

-- get score of each view 
view_score = torch.load('./score_prediction_logs/view_score.t7') 
view_score_train = view_score['train'] 
view_score_test = view_score['test'] 

-- Extract train set features: that is the removed layer are the network we pre-trained 
-- and not fine-tuned a afterwards but as a feature extractor 
train_data = {}
train_label = {}
train_cnt = 1
-- extract features of train dataset of modelnet40.h5
-- in order to extract features of all the images the dataset 
-- we use one images per iteration
for t = 1, loader.num_images_split['train'] do 
        xlua.progress(t, loader.num_images_split['train']) 
        -- data.images_mv: 1 x 12 x 1 x 224 x 224 
        local data = loader:getBatch(1, 'train')
        
        if true then 
            softmax = nn.SoftMax()
            view_score_t = softmax:forward(view_score_train:select(1, loader.idx_list_split['train'][t])) 
        end 

        -- squeeze only first dimensiton 
        -- 12 x 1 x 224 x 224 
        local input = data.images_mv:squeeze(1)
        
        ---------------------------------
        --create an RGB images 
        --------------------------------
        -- ##
        input = input:expand(input:size(1), 3, input:size(3), input:size(4))
        -- #

        -- preprocess input 
        input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augment
        local target = data.label 
        model:forward(input)
        local features = model:get(layer_idx).output

        -- max pool over the features of 12 views 
        -- features: 12 x 1696 x 14 x 14 
        if false  then 
            local max_pooled_feature = torch.max(features, 1) -- 12 view 
        
            train_data[train_cnt] = max_pooled_feature 
        else 
            view_score_t = view_score_t:view(12, 1, 1, 1)  
            view_score_t = view_score_t:expand(12, features:size(2), features:size(3), features:size(4)):cuda() 
            features = features:cmul(view_score_t) 

            local atten_pooled_features = torch.sum(features, 1)

            train_data[train_cnt] = atten_pooled_features 
        end 

        train_label[train_cnt] = target 
        train_cnt = train_cnt + 1 
        if t == loader.num_images_split['train'] and data.wrapped then
            print('last image, and wrapped is True')
        end 
end 

-- Extract train set features: that is the removed layer are the network we pre-trained 
-- and not fine-tuned a afterwards but as a feature extractor 
test_data = {}
test_label = {}
test_cnt = 1
-- extract features of train dataset of modelnet40.h5
-- in order to extract features of all the images the dataset 
-- we use one images per iteration 
for t = 1, loader.num_images_split['test'] do 
        xlua.progress(t, loader.num_images_split['test']) 
        -- data.images_mv: 1 x 12 x 1 x 224 x 224 
        local data = loader:getBatch(1, 'test')

        if true then 
            softmax = nn.SoftMax()
            view_score_t = softmax:forward(view_score_test:select(1, loader.idx_list_split['test'][t])) 
        end

        -- squeeze only first dimensiton 
        -- 12 x 1 x 224 x 224 
        local input = data.images_mv:squeeze(1)
        
        ---------------------------------
        --create an RGB images, because DenseNet is trained on ImageNet RGB images 
        --------------------------------
        -- ##
        input = input:expand(input:size(1), 3, input:size(3), input:size(4)) -- 12 x 3 x 224 x 224
        -- ##
        
        -- preprocess input 
        input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augmen

        local target = data.label 
        model:forward(input)
        local features = model:get(layer_idx).output

        if false then 
            local max_pooled_feature = torch.max(features, 1) -- 12 view 
            test_data[test_cnt] = max_pooled_feature 
        else  
            view_score_t = view_score_t:view(12, 1, 1, 1)  
            view_score_t = view_score_t:expand(12, features:size(2), features:size(3), features:size(4)):cuda() 
            features = features:cmul(view_score_t) 

            local atten_pooled_features = torch.sum(features, 1)

            test_data[test_cnt] = atten_pooled_features 
        end 

        test_label[test_cnt] = target 
        test_cnt = test_cnt + 1 
        if t == loader.num_images_split['test'] and data.wrapped then
            print('last image, and wrapped is True')
        end 
end 

collectgarbage() 

print('number of training 3D models: ' .. #train_data)
print('number of 3D test models: ' .. #test_data)

print('Starting to train multi-orientation pooling ...')

-- config for SGD solver
--[[ original configuration 
optimState = {
    learningRate = opt.learning_rate,
    weightDecay = 0.00005,
    momentum = 0.9,
    learningRateDecay = 1e-7,
}
--]] 


optimState = {
  learningRate = opt.learningRate,
  weightDecay = 0.00005,
  momentum = 0.9,
  nesterov = true,
  dampening = 0.0,
}
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
model = model_pool
epoch = nil
function train()
    model:training()
    epoch = epoch or 1 -- if epoch not defined, assign it as 1

    if epoch < 82 then  
        optimState.learningRate = 1e-3   
        -- according to original paper, we will divide the learning rate by 10 at 80 epochs and 120 epochs
    elseif epoch < 122 then 
        optimState.learningRate = 1e-4  -- learningRate = 0.01 afterwards
    else 
        optimState.learningRate = 1e-5   -- learningRate = 0.001 afterwards
    end 
    -- original way to change learning rate 
    -- if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end

    local tic = torch.tic()
    local filesize = #train_data
    local targets = torch.CudaTensor(opt.batch_size)
    local indices = torch.randperm(filesize):long():split(opt.batch_size)
    -- remove last mini-batch so that all the batches have equal size
    indices[#indices] = nil 

    for t, v in ipairs(indices) do 
        xlua.progress(t, #indices) 

        local inputs = train_data[v[1]]
        for i = 2,opt.batch_size do
            inputs = torch.cat(inputs, train_data[v[i]],1)
        end 

        for i = 1,opt.batch_size do
            targets[i] = train_label[v[i]]
        end
        -- targets: 64 Tensor 

        -- a function that takes single input and return f(x) and df/dx
        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do) -- gradParameters in model have been updated

            if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
                confusion:batchAdd(outputs[#outputs], targets)
            else
                confusion:batchAdd(outputs, targets)    
            end

            return f, gradParameters
        end

        -- use SGD optimizer: parameters as input to feval will be updated
        optim.sgd(feval, parameters, optimState)
    end
    
    confusion:updateValids()
    print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
            confusion.totalValid * 100, torch.toc(tic)))

    train_acc = confusion.totalValid * 100

    confusion:zero()
    epoch = epoch + 1
end

----------------------------------------
-- Test routine
--
average_class_acc = 0
function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    
    local filesize = #test_data
    local indices = torch.randperm(filesize):long():split(opt.batch_size)

    for t, v in ipairs(indices) do
        -- v: an indices batch 
        local inputs = test_data[v[1]]
        
        for i = 2,v:size(1) do
            inputs = torch.cat(inputs, test_data[v[i]],1)
        end

        local targets = torch.CudaTensor(v:size(1))
        for i = 1,v:size(1) do
            targets[i] = test_label[v[i]]
        end
        
        local outputs = model:forward(inputs)

        if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
            confusion:batchAdd(outputs[#outputs], targets)
        else
            confusion:batchAdd(outputs, targets)    
        end

    end

    confusion:updateValids()
    print('average instance accuracy (test set):', confusion.totalValid * 100)
    print('average class accuracy (test set):', confusion.averageValid*100) 

    -- logging test result to txt and html files
    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add{train_acc, confusion.totalValid * 100, confusion.averageValid*100}
        testLogger:style{'-','-','-'}
        testLogger:plot()

        local base64im
        do
          os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
          os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
          local f = io.open(opt.save..'/test.base64')
          if f then base64im = f:read'*all' end
        end

        local file = io.open(opt.save..'/report.html','w')
        file:write(([[
        <!DOCTYPE html>
        <html>
        <body>
        <title>%s - %s</title>
        <img src="data:image/png;base64,%s">
        <h4>optimState:</h4>
        <table>
        ]]):format(opt.save,epoch,base64im))
        for k,v in pairs(optimState) do
          if torch.type(v) == 'number' then
            file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
          end
        end
        file:write'</table><pre>\n'
        file:write(tostring(confusion)..'\n')
        file:write(tostring(model)..'\n')
        file:write'</pre></body></html>'
        file:close()
    end

    -- save model every 10 epochs
    if average_class_acc < confusion.averageValid * 100 then 
        average_class_acc = confusion.averageValid * 100 
        local filename = paths.concat(opt.save, 'checkpoint.t7')
        print('==> saving model to '..filename)
        torch.save(filename, model:clearState())
    end 

    print('best average class accuracy: ', average_class_acc) 
    confusion:zero()
end

----------------------------------------
-- Start training
----------------------------------------
for e = 1,opt.max_epoch do
    train()
    collectgarbage()
    test()
    collectgarbage() 
end  
