require 'hdf5' 

-- Loading Data from .h5 data 
local DataLoader = torch.class('DataLoader') 

function DataLoader:__init(opt) 
	-- open the hdf5 file 
	print('DataLoader loading h5 file: ', opt.input_h5) 
    self.h5_file = hdf5.open(opt.input_h5, 'r') 

    -- exact image size 
    -- can make sure each views in the image are ordered, because we sample along num3DModels 
    -- num3DModels x 12 x 1 x 224 x 224, note that 12 views per 3d models 
    local train_images_size = self.h5_file:read('/train/data'):dataspaceSize() 
    self.num_views = train_images_size[2] 
    self.num_channels = train_images_size[3] 
    self.img_height = train_images_size[4] 
    self.img_width = train_images_size[5] 
    assert(self.img_height == self.img_width, 'image height and width must be equal')

    self.iterators = {}
    self.num_images_split = {} 
    self.idx_list_split = {} 

    -- Training Dataset 
    self.num_images_split['train'] = train_images_size[1] 
    self.idx_list_split['train'] = torch.randperm(self.num_images_split['train']) 
    self.iterators['train'] = 1 
    
    -- Testing Dataset  
    local test_images_size = self.h5_file:read('/test/data'):dataspaceSize() 
    self.num_images_split['test'] = test_images_size[1]  
    self.idx_list_split['test'] = torch.randperm(self.num_images_split['test']) 
    self.iterators['test'] = 1 

end 

function DataLoader:resetIterator(split) 
    self.iterators[split] = 1 
end 

function DataLoader:permute() 
    -- we can call it to random permute training data 
    self.idx_list_split['train'] = torch.randperm(self.num_images_split['train']) 
end 

function DataLoader:getBatch(batch_size, split)  

    local split = split or 'train' 
    -- 12 views 
    local img_batch_raw = torch.ByteTensor(batch_size, 12, 1, 224, 224)
    local label_batch = torch.ByteTensor(batch_size) 
    local wrapped = false 

    for i = 1, batch_size do 
        local ri  = self.iterators[split] 
        local ri_next = ri + 1 -- increment iterators 
        if ri_next > self.num_images_split[split] then ri_next = 1; wrapped = true end -- wrap back around 
        self.iterators[split] = ri_next

        ix = self.idx_list_split[split][ri] 
        assert(ix ~= nil, 'ix must be not nil')
        -- fetch the image from h5 
        local img = self.h5_file:read(split .. '/data'):partial({ix, ix}, {1, 12}, {1, self.num_channels}, {1, self.img_height}, {1, self.img_width}) 
        img_batch_raw[i] = img 
        
        -- fetch label 
        local label = self.h5_file:read(split..'/label'):partial({ix, ix}) 
        label_batch[i] = label 
    end 
    
    local data = {} 
    data.images_mv = img_batch_raw 
    data.label = label_batch 
    data.wrapped = wrapped 

    return data 
end

function DataLoader:computeMeanStd() 
    -- we use the training set to compute mean and std for the dataset 
    print('computing mean and std of the training dataset .. ')
    local mean = torch.Tensor(1):zero() -- because for Grey image, 1 channel 
    local std = torch.Tensor(1):zero()

    for i = 1, self.num_images_split['train'] do
        if i % 100 == 0 then print('processed: ' .. i .. '/' .. self.num_images_split['train']) end 
        local x = self:getBatch(1, 'train')
        mean[1] = mean[1] + x.images_mv:float():mean()
        std[1] = std[1] + x.images_mv:float():std()  
    end 
    mean:div(self.num_images_split['train']) 
    std:div(self.num_images_split['train']) 
    print('mean: ' .. mean[1]) 
    print('std: ' .. std[1]) 
    self.mean = mean 
    self.std = std 
    print('done')
    return mean, std 
end 

