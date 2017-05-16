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
cmd:text('view score getter based on trained view score prediction net')

cmd:text()
cmd:text('Options')

cmd:option('-silent', false, 'print opt to the screen?')
cmd:option('-seed', 1234, 'print opt to the screen?')
cmd:option('-save', './score_prediction_logs', 'subdirectory to save logs') 
cmd:option('-backend', 'cudnn', 'whether we use cudnn backend, nn | cudnn')
cmd:option('-gpu_id', 1, 'GPU index')
cmd:option('-input_h5', 'data/modelnet40.h5', 'h5 file that contains modelnet40 dataset ')
cmd:option('-view_score_prediction_net', './score_prediction_logs/checkpoint.t7', 'path to view score prediction network')

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

model = torch.load(opt.view_score_prediction_net)
if opt.gpu_id > 0 then  
    model = model:cuda() 
end 

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

print('Starting to annotate each view  with score  using score prediction network ...')

confusion = optim.ConfusionMatrix(40)
confusion:zero()

----------------------------------------
-- Annotate views for each shape in the Training Set 
----------------------------------------
model:evaluate() 

-- save score 
view_score={}
local tic = torch.tic()

idx_list_train = loader.idx_list_split['train']
vscore_map_train = torch.Tensor(loader.num_images_split['train'], 12):zero() 
for t = 1, loader.num_images_split['train'] do 
    xlua.progress(t, loader.num_images_split['train'])
    -- get views of one 3d shape (12 views) 
    data = loader:getBatch(1, 'train') 
    --##
    idx = idx_list_train[t]
    -- ## 

    -- 1 x 12 x 1 x 224 x 224 
    inputs = data.images_mv
    -- 12 
    inputsV_12x = inputs:view(1*12, inputs:size(3), inputs:size(4), inputs:size(5))
    inputsV_12x = inputsV_12x:expand(inputsV_12x:size(1), 3, inputsV_12x:size(3), inputsV_12x:size(4))
    inputsV_12x = net_utils.prepro(inputsV_12x, false, modelnet40_mean, opt.gpu_id>=0)
    targets = data.label 

    targets_12x = targets:view(1, 1):expand(1, 12):contiguous():view(-1)
    if opt.gpu_id >= 0 then 
        targets_12x = targets_12x:cuda()
    end 

    outputs = model:forward(inputsV_12x)

    if torch.type(outputs) == 'table' then -- multiple outputs, take the last one 
        confusion:batchAdd(outputs[#outputs], targets_12x) 
    else 
        confusion:batchAdd(outputs, targets_12x)
    end 

    -- log likelihood to likelihood (prob.)
    -- 12 x 40 
    outputs:exp()
    -- take out probs and write to vscore_map_train
    -- targets: 12 
    vscore_map_train[idx] = outputs:select(2, targets_12x[1]):double()

end 

confusion:updateValids()
print(('Train accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
train_acc = confusion.totalValid * 100
confusion:zero()
 
view_score['train'] = vscore_map_train 

----------------------------------------
-- Annotate views for each shape in the Test Set 
----------------------------------------
tic = torch.tic()

idx_list_test = loader.idx_list_split['test']
vscore_map_test = torch.Tensor(loader.num_images_split['test'], 12):zero() 
for t = 1, loader.num_images_split['test'] do 
    xlua.progress(t, loader.num_images_split['test'])
    -- get views of one 3d shape (12 views) 
    data = loader:getBatch(1, 'test') 
    -- ##
    idx = idx_list_test[t]
    -- ## 

    -- 1 x 12 x 1 x 224 x 224 
    inputs = data.images_mv
    -- 12 
    inputsV_12x = inputs:view(1*12, inputs:size(3), inputs:size(4), inputs:size(5))
    inputsV_12x = inputsV_12x:expand(inputsV_12x:size(1), 3, inputsV_12x:size(3), inputsV_12x:size(4))
    inputsV_12x = net_utils.prepro(inputsV_12x, false, modelnet40_mean, opt.gpu_id>=0)
    targets = data.label 

    targets_12x = targets:view(1, 1):expand(1, 12):contiguous():view(-1)
    if opt.gpu_id >= 0 then 
        targets_12x = targets_12x:cuda()
    end 

    outputs = model:forward(inputsV_12x)

    if torch.type(outputs) == 'table' then -- multiple outputs, take the last one 
        confusion:batchAdd(outputs[#outputs], targets_12x) 
    else 
        confusion:batchAdd(outputs, targets_12x)
    end 

    -- log likelihood to likelihood (prob.)
    outputs:exp()
    -- take out probs and write to vscore_map_test
    -- targets: 12 
    -- print(outputs:select(2, targets_12x[1]))
    vscore_map_test[idx] = outputs:select(2, targets_12x[1]):double()
    
end 

confusion:updateValids()
print(('Test accuracy: '..'%.2f'..' %%\t time: %.2f s'):format(
        confusion.totalValid * 100, torch.toc(tic)))
confusion:zero()

view_score['test'] = vscore_map_test

torch.save('./score_prediction_logs/view_score.t7', view_score)

