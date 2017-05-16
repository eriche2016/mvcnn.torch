require 'nn'
require 'cunn'
require 'cudnn'

-- extra dependencies 
require '../misc/DataLoader.lua'

local net_utils =  require '../misc/net_utils.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Generate features from pretrained models')

cmd:text()
cmd:text('Options')
cmd:option('-silent', false, 'print opt to the screen?')
cmd:option('-seed', 1234, 'print opt to the screen?')
cmd:option('-gpu_id', 1, 'GPU index')
cmd:option('-cnn_model', '../mv_logs/checkpoint.t7', '')
cmd:option('-input_h5', '../data/modelnet40.h5', 'h5 file that contains modelnet40 dataset ')
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


-- loading dataset 
loader = DataLoader(opt) 

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

print('Loading pretrained model...')
checkpoint = torch.load(opt.cnn_model)
model_pool = checkpoint.model_pool
model_before_pool = checkpoint.model_before_pool
Rotate_Kernels = checkpoint.Rotate_Kernels -- already cudatensor 


if opt.gpu_id >= 0 then 
   model_pool = model_pool:cuda() 
   model_before_pool:cuda() 
end 
print('model_before_pool: ')
print(model_before_pool)
model_before_pool:evaluate() 
print('model_pool: ')
print(model_pool)
model_pool:evaluate() 

local train_emb = torch.Tensor(loader.num_images_split['train'], 512)
local train_label = torch.Tensor(loader.num_images_split['train'])
local test_emb = torch.Tensor(loader.num_images_split['test'], 512)
local test_label = torch.Tensor(loader.num_images_split['test'])

for t = 1, loader.num_images_split['train'] do
	xlua.progress(t, loader.num_images_split['train']) 
	local data = loader:getBatch(1, 'train')

	-- forward data to get embeddings 
	-- 12 x 1 x 224 x 224 
    local input = data.images_mv:squeeze(1)
    input = input:expand(input:size(1), 3, input:size(3), input:size(4))
    input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augment
    local target = data.label 
    features = model_before_pool:forward(input) 

    norm_view_module = nn.Sequential() 
    norm_view_module:add(nn.Normalize(1))
    norm_view_module:add(nn.View(-1, features:size(2), 12, 1)) 
    if opt.gpu_id >= 0 then 
        norm_view_module:cuda() 
    end

    features_v = norm_view_module:forward(features)
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
    -- features_max_pool = max_module:forward(features_out)   
    -- ##
    -- merge 
    -- merged_features = merge_features_net:forward({features, features_max_pool}) 
    -- ## 

    -- forward to model_pool 
    outputs = model_pool:forward(features_max_pool) 

    -- take features_max_pool 
    embedding = features_max_pool 

    -- set embedding and label 
    train_emb:narrow(1, t, 1):copy(embedding)
    train_label:narrow(1, t, 1):copy(target) 
end
print('saving training embedding data ')
torch.save('./embed_data/train_emb.t7', {train_emb, train_label})

----------------------------------------------------------------
-- test data 
----------------------------------------------------------------
for t = 1, loader.num_images_split['test'] do
    xlua.progress(t, loader.num_images_split['test']) 
    local data = loader:getBatch(1, 'test')

    -- forward data to get embeddings 
    -- 12 x 1 x 224 x 224 
    local input = data.images_mv:squeeze(1)
    input = input:expand(input:size(1), 3, input:size(3), input:size(4))
    input = net_utils.prepro(input, false, modelnet40_mean, opt.gpu_id>=0) -- preprocess in place, and donot augment
    local target = data.label 
    features = model_before_pool:forward(input) 

    norm_view_module = nn.Sequential() 
    norm_view_module:add(nn.Normalize(1))
    norm_view_module:add(nn.View(-1, features:size(2), 12, 1)) 
    if opt.gpu_id >= 0 then 
        norm_view_module:cuda() 
    end

    features_v = norm_view_module:forward(features)
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
    -- features_max_pool = max_module:forward(features_out)   
    -- ##
    -- merge 
    -- merged_features = merge_features_net:forward({features, features_max_pool}) 
    -- ## 

    -- forward to model_pool 
    outputs = model_pool:forward(features_max_pool) 

    -- take features_max_pool 
    embedding = features_max_pool 

    -- set embedding and label 
    test_emb:narrow(1, t, 1):copy(embedding)
    test_label:narrow(1, t, 1):copy(target) 
end

print('saving testing embedding data ')
torch.save('./embed_data/test_emb.t7', {test_emb, test_label})
print('done!!')
