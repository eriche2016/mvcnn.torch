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

