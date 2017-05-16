require 'misc.DataLoader'  
require 'image'

opt = {} 
opt.input_h5 = './data/modelnet40.h5' 

loader = DataLoader(opt) 

-- get a batch i
-- during testting, in order to test all the images 
-- we must need ensure that the batch_size is a divisor of 
-- the number of images in the test set 
k = 0 
while k < 1 do 
    k = k + 1 
    data_batch = loader:getBatch(1, 'test')
    print(k .. '-th batch')

    if data_batch.wrapped then 
        break 
    end 

end

mean, std = loader:computeMeanStd()
-- check figure 
image.save('./test.jpg', data_batch.images_mv[1][1])
