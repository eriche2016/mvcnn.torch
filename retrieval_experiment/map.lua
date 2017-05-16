require 'xlua'

torch.setnumthreads(2)
local metric = paths.dofile('metric.lua')

-- loading training embed
local train_featurevec_label = torch.load('./embed_data/train_emb.t7') 
train_featurevec = train_featurevec_label[1]
train_label = train_featurevec_label[2] 

-- loading test embed
local test_featurevec_label = torch.load('./embed_data/test_emb.t7') 
test_featurevec = test_featurevec_label[1] 
test_label = test_featurevec_label[2] 


print('train_featurevec size: ', train_featurevec:size()) 
print('test_featurevec size: ', test_featurevec:size()) 
local distance_matrix = torch.FloatTensor(test_featurevec:size(1), train_featurevec:size(1)) 
local labels_matrix = torch.FloatTensor(test_featurevec:size(1), train_featurevec:size(1)) 

local time = sys.clock() 
print('start timing') 

for i = 1, test_featurevec:size(1) do 
    xlua.progress(i, test_featurevec:size(1))
    for j = 1, train_featurevec:size(1) do 
        distance_matrix[i][j] = torch.dist(test_featurevec[i], train_featurevec[j], 2) 
        labels_matrix[i][j] = (test_label[i] == train_label[j]) and 1 or -1 
    end 
end 

time = sys.clock() - time 
local prec, recall, ap, map, ind = metric.precisionrecall(distance_matrix, labels_matrix, 0.1, test_featurevec) 
time = time / (test_featurevec:size(1) * train_featurevec:size(1)) 

print('<trainer> time to compute distance between 2 vectors = ' .. (time*1000) .. ' ms')
print('mean average precision %: ', map * 100) 
print('')
