require 'misc.PrepMVDatasetFolder' 


opt = {} 
opt.folder_path = './data/modelnet40v1' 
opt.img_size = 224
-- we have already convert data in folders to .h5 file 
data = PrepMVDatasetFolder(opt)
 
