local LBP, parent = torch.class('nn.LBP', 'nn.Module') 

function sigmoid(x) 
    return 1/(1 + torch.exp(-x))
end 

function gradsigmoid(x) 
    local sigx = sigmoid(x) 
    return sigx * (1 - sigx) 
end 

function top(x, y)          return x-1, y       end
function topright(x, y)     return x-1, y+1     end
function right(x, y)        return x,   y+1     end
function bottomright(x, y)  return x+1, y+1     end
function bottom(x, y)       return x+1, y       end
function bottomleft(x, y)   return x+1, y-1     end
function left(x, y)         return x,   y-1     end
function topleft(x, y)      return x-1, y-1     end

function lbp:updateOutput(input)
  
  self.output = torch.Tensor(input:size(1), input:size(2) - 2, input:size(3) - 2) 
  
  --calculate lbp for each non-corner pixel
  for depth = 1, input:size(1) do
    for index1 = 2 , input:size(2) - 1 do
      for index2 = 2 , input:size(3) - 1 do
        local valueInFocus = input[depth][index1][index2]
        
        local outputval = 0;
        
        --top
        row, col = top(index1, index2)
        outputval = outputval + (bit.lshift(1,0) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --topright
        row, col = topright(index1, index2)
        outputval = outputval + (bit.lshift(1,1) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --right
        row, col = right(index1, index2)
        outputval = outputval + (bit.lshift(1,2) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --bottomright
        row, col = bottomright(index1, index2)
        outputval = outputval + (bit.lshift(1,3) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --bottom
        row, col = bottom(index1, index2)
        outputval = outputval + (bit.lshift(1,4) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --bottomleft
        row, col = bottomleft(index1, index2)
        outputval = outputval + (bit.lshift(1,5) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --left
        row, col = left(index1, index2)
        outputval = outputval + (bit.lshift(1,6) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --topleft
        row, col = topleft(index1, index2)
        outputval = outputval + (bit.lshift(1,7) * sigmoid(valueInFocus - input[depth][row][col]))
        
        --determine ~lbp value
        self.output[depth][index1 - 1][index2 - 1] = outputval
      end
    end
  end

  return self.output
end

function lbp:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):fill(0)
  
  for depth = 1, input:size(1) do
  for index1 = 1, input:size(1) do
    for index2 = 1, input:size(2) do
      local gradientVal = 0
      local valueInFocus = input[index1][index2]
      
      --w.r.t, bottom
      row, col = bottom(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,0) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, bottomleft
      row, col = bottomleft(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,1) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, left
      row, col = left(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,2) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, topleft
      row, col = topleft(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,3) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, top
      row, col = top(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,4) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, topright
      row, col = topright(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,5) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end    
      
      --w.r.t, right
      row, col = right(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,6) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --w.r.t, bottomright
      row, col = bottomright(index1, index2)
      if(row > 1 and col > 1 and row < input:size(1) and col < input:size(2)) then
        gradientVal = gradientVal + (bit.lshift(1,7) * -gradsigmoid(input[row][col] - valueInFocus)
                                      * gradOutput[row-1][col-1])
      end
      
      --if the lbp has been found for this particular pixel, then calculate gradient
      if(index1 > 1 and index2 > 1 and index1 < input:size(1) and index2 < input:size(2)) then
        --top
        local localGradient = 0
        row, col = top(index1, index2)
        localGradient = localGradient + (bit.lshift(1,0) * gradsigmoid(valueInFocus - input[row][col]))
        
        --topright
        row, col = topright(index1, index2)
        localGradient = localGradient + (bit.lshift(1,1) * gradsigmoid(valueInFocus - input[row][col]))
        
        --right
        row, col = right(index1, index2)
        localGradient = localGradient + (bit.lshift(1,2) * gradsigmoid(valueInFocus - input[row][col]))
        
        --bottomright
        row, col = bottomright(index1, index2)
        localGradient = localGradient + (bit.lshift(1,3) * gradsigmoid(valueInFocus - input[row][col]))
        
        --bottom
        row, col = bottom(index1, index2)
        localGradient = localGradient + (bit.lshift(1,4) * gradsigmoid(valueInFocus - input[row][col]))
        
        --bottomleft
        row, col = bottomleft(index1, index2)
        localGradient = localGradient + (bit.lshift(1,5) * gradsigmoid(valueInFocus - input[row][col]))
        
        --left
        row, col = left(index1, index2)
        localGradient = localGradient + (bit.lshift(1,6) * gradsigmoid(valueInFocus - input[row][col]))
        
        --topleft
        row, col = topleft(index1, index2)
        localGradient = localGradient + (bit.lshift(1,7) * gradsigmoid(valueInFocus - input[row][col]))
        
        localGradient = localGradient * gradOutput[index1-1][index2-1]
        gradientVal = gradientVal + localGradient
      end
      
      self.gradInput[index1][index2] = gradientVal
    end
  end
end

  return self.gradInput
end

