require 'lfs' 
require 'image'
require 'paths'
require 'hdf5' 
require 'xlua'


local PrepMVDatasetFolder = torch.class('PrepMVDatasetFolder') 

function PrepMVDatasetFolder:__init(opt) 
    print('PrepMVDatasetFolder loading images from folder: ', opt.folder_path) 
    self.num_channels = 1

    self.class_names = {} 
    
    for folder_name in paths.iterdirs(opt.folder_path) do 
        -- note that foldername is its class name 
        table.insert(self.class_names, folder_name)
    end 
    
    -- make h5 file 
    self.h5file = hdf5.open('./data/modelnet40.h5', 'w') 
    
        -- read in all filenames from the folder 
    -- print('listting all the images in directory: ' .. opt.folder_pat)
    local function isImage(f) 
        local supportedExt = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM'} 
        for _, ext in pairs(supportedExt) do 
            local _, end_idx = f:find(ext) 
            if end_idx and end_idx == f:len() then 
                return true 
            end 
        end 
        return false 
    end

    local function is_in_Table(f_test, f_table) 
        for i, f in pairs(f_table)  do 
            if f_test == f then 
                return true 
            end 
        end 
        return false 
    end 
    -- self.files = {} 
    self.cls2idx = {} 
    -- loading data 
    self.files = {} 
    
    for i, split in ipairs({'train', 'test'}) do 
        local files_split = {} 
        -- loading training data 
        for k, cls_name in ipairs(self.class_names) do  
            local class_path = opt.folder_path .. '/' .. cls_name .. '/'.. split 
            -- listing all the images in the directory 
            print('listting all the images in directory: ' ..class_path) 
            
            for file in paths.files(class_path, isImage) do 
                -- 12 views, file format: bowl/test/bowl_000000404_011.jpg
                local index = string.find(file, "_[^_]*$")
                -- bowl/test/bowl_000000404_
                local full_path = path.join(class_path, file:sub(1, index))

                if not is_in_Table(full_path, files_split) then
                    table.insert(files_split, full_path)
                end 

            end  
            self.cls2idx[cls_name] = k  
        end 
        
        -- loading data 
        self.files[split] = files_split 
        print('there are ' .. #self.files[split] .. ' 3d models for ' .. split)

        -- each models with 12 views 
        local Split_Data = torch.ByteTensor(#self.files[split], 12, 1, opt.img_size, opt.img_size):zero() 
        local Split_Label = torch.ByteTensor(#self.files[split]):zero() 
         
        -- each 3d model is rendered with 12 views, so ...
        for i, file in ipairs(files_split) do 
            
            xlua.progress(i, #files_split)

            dic_end_idx = {'001', '002', '003', '004','005', '006','007', '008','009', '010','011', '012'}
            for j, end_idx in ipairs(dic_end_idx) do 
                local file_path = file ..end_idx.. '.jpg'
                -- 1 channels 
                local img = image.load(file_path, self.num_channels , 'byte') 
                if img:dim() == 2 then img = img:resize(1, img:size(1), img:size(2)) end 
                if img:size(2) > opt.img_size or img:size(3) > opt.img_size then 
                    img = image.scale(img, opt.img_size) 
                end
                Split_Data[{{i}, {j}}] = img 
            end  
            
            -- ./data/modelnet40v1/keyboard/train/**.png
            local idx1 = string.find(file, 'modelnet40v1') + 13 -- 13 is number of characters in modelnet 
            local idx2 = string.find(file, split) -2  -- split may be 'train' or 'test', -2 will mode pointer 2 chars ahead(ex: '/', 't')
            local cls_name = file:sub(idx1, idx2)
            Split_Label[i] = self.cls2idx[cls_name]
        end 
        -- load images and write it into h5 file
        print('writting data to h5 file ')
        self.h5file:write(split .. '/data', Split_Data)
        print('writing label to h5 file')
        self.h5file:write(split .. '/label', Split_Label) 
    end 
end 
