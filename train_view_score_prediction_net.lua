require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'xlua'
local nninit = require 'nninit'

-- extra dependencies 
require 'misc.DataLoader'
local net_utils =  require 'misc.net_utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train view score prediction net')

cmd:text()
cmd:text('Options')

cmd:option('-silent', false, 'print opt to the screen?')
cmd:option('-seed', 1234, 'print opt to the screen?')
cmd:option('-save', './score_prediction_logs', 'subdirectory to save logs') 
cmd:option('-batch_size', 2, 'batch size, will expand to 12x')
cmd:option('-learning_rate', 1e-3, 'leanring rate')
cmd:option('-learning_rate_decay', 1e-7, 'learning_rate_decay')
cmd:option('-weight_decay', 5e-4, 'weight decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-epoch_step', 20, 'epoch step, we decrease the learning rate by half')

cmd:option('-backend', 'cudnn', 'whether we use cudnn backend, nn | cudnn')
cmd:option('-gpu_id', 0, 'GPU index')
cmd:option('-max_epoch', 100, 'maximum number of epochs')
cmd:option('-cnn_model', './models/pre_trained_net/imagenet_vgg_m_optnet.t7', 'path to cnn model that will be the base net for mv cnn: densenet-201.t7')
cmd:option('-input_h5', 'data/modelnet40.h5', 'h5 file that contains modelnet40 dataset ')
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
if string.len(opt.init_from) > 0 then 
    print('loading model from checkpoint' .. opt.init_from)
    model = torch.load(opt.init_from)
else 
     -- choose which net 
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
    elseif false then 
        local model_raw = torch.load(opt.cnn_model):cuda() 
        model_raw:remove(22)
        model_raw:remove(21) 
        
        model = model_raw
        -- add additional layer 
        model:add(nn.Linear(4096, 40):init('weight', nninit.xavier, {dist = 'normal', gain = 1.1})) 
        opt.grad_clip_or_not = true 
    else 
        model = nn.Sequential() 
        model:add(nn.SpatialConvolution(3, 96, 11, 11, 4, 4))
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
        -- when training from scratch, we donot need to do gradient clipping stuff  
        opt.grad_clip_or_not = false 
    end 
end 

if opt.gpu_id >= 0 then 
    model = model:cuda()
end 

-- set criterion
-- unused, criterion = dofile('torch_models/'..opt.model..'.lua')
-- assert(#model == #unused) -- check for consistency

criterion = nn.CrossEntropyCriterion():cuda()
parameters, gradParameters = model:getParameters()

print('Loading data...')

--------------------------------------
-- compute mean and std for the dataset, which will be used in net_utils.prepro
--------------------------------------
local modelnet40_mean_tensor, modelnet40_std_tensor= torch.Tensor{223.03979492188},  torch.Tensor{57.232696533203} --loader:computeMeanStd()
modelnet40_mean = modelnet40_mean_tensor[1]
use_mean_only = true 
-- currenly, std is useless in net_utils.prepro
-- ## 
if use_mean_only then -- just reset it to 1  
    modelnet40_std = 1 
end 


-- ######################################################################################

print('Starting to train view score prediction network ...')

-- config for SGD solver
optimState = {
    learningRate = opt.learning_rate,
    weightDecay = 0.00005,
    momentum = 0.9,
    learningRateDecay = 1e-7,
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

epoch = nil 
function train()
    model:training()
    epoch = epoch or 1 -- if epoch not defined, assign it as 1
    
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
    local tic = torch.tic()

    for t = 1, loader.num_images_split['train'], opt.batch_size do  
        xlua.progress(t, loader.num_images_split['train']) 
        -- get a batch of input data 
        data = loader:getBatch(opt.batch_size, 'train')
        -- bz x 12 x 1 x 224 x 224 
        inputs = data.images_mv
        -- inputsV: {V[1][1], V[1][2], ...V[1][12], V[2][1], ...}
        -- (bz*12) x 1 x 224 x 224 
        inputsV_12x = inputs:view(opt.batch_size*12, inputs:size(3), inputs:size(4), inputs:size(5))
        -- preprocess inputsV_12x 
        -- bz * 12 x 1 x 224 x 224 
        inputsV_12x = inputsV_12x:expand(inputsV_12x:size(1), 3, inputsV_12x:size(3), inputsV_12x:size(4)) 
        inputsV_12x = net_utils.prepro(inputsV_12x, false, modelnet40_mean, opt.gpu_id>=0)

        -- now we would like to split the data along view dimensions 
        -- bz 
        targets = data.label
        -- we expand the targets to 12x 
        targets_12x = targets:view(opt.batch_size, 1):expand(opt.batch_size, 12):contiguous():view(-1)
        if opt.gpu_id >= 0 then 
            targets_12x = targets_12x:cuda()
        end 

        -- a function that takes single input and return f(x) and df/dx 
        local feval = function(x) 
            if x ~= parameters then parameters:copy(x) end 
            gradParameters:zero() 
            local outputs = model:forward(inputsV_12x)
            local f = criterion:forward(outputs, targets_12x)
            
            local df_do = criterion:backward(outputs, targets_12x)
            model:backward(inputsV_12x, df_do) 

            if torch.type(outputs) == 'table' then -- multiple outputs, take the last one 
                confusion:batchAdd(outputs[#outputs], targets_12x) 
            else 
                confusion:batchAdd(outputs, targets_12x)
            end 

            -- when using imagenet_vgg_m, we use nned to clip the gradient otherwise, it 
            -- will not converge 
            if opt.grad_clip_or_not then 
                -- gradient clip 
                -- average gradient 
                gradParameters:div(opt.batch_size * 12)
                gradParameters:clamp(-0.1, 0.1)
            end 
            
            return f, gradParameters
        end

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
    
    for t = 1, loader.num_images_split['test'] do 
        xlua.progress(t, loader.num_images_split['test'])
        -- get a batch of input data 
        data = loader:getBatch(1, 'test')
        -- 1 x 12 x 1 x 224 x 224 
        inputs = data.images_mv
        -- inputsV: {V[1][1], V[1][2], ...V[1][12], V[2][1], ...}
        -- (bz*12) x 1 x 224 x 224 
        inputsV_12x = inputs:view(1*12, inputs:size(3), inputs:size(4), inputs:size(5))
        -- preprocess inputsV_12x 
        -- bz * 12 x 1 x 224 x 224 
        inputsV_12x = inputsV_12x:expand(inputsV_12x:size(1), 3, inputsV_12x:size(3), inputsV_12x:size(4)) 
        inputsV_12x = net_utils.prepro(inputsV_12x, false, modelnet40_mean, opt.gpu_id>=0)

        -- now we would like to split the data along view dimensions 
        -- bz 
        targets = data.label
        -- we expand the targets to 12x 
        targets_12x = targets:view(1, 1):expand(1, 12):contiguous():view(-1)
        if opt.gpu_id >= 0 then 
            targets_12x = targets_12x:cuda()
        end 

        local outputs = model:forward(inputsV_12x)

        if torch.type(outputs) == 'table' then -- multiple outputs, take the last one
            confusion:batchAdd(outputs[#outputs], targets_12x)
        else
            confusion:batchAdd(outputs, targets_12x)    
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
