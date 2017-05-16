require 'nn';
require 'CircularConvolution'

model = nn.CircularConvolution() 

weights = torch.rand(12) 

print('start')

model:forward({torch.rand(80000*12), weights})
print('done')
 
