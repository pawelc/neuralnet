--MatrixLayer is the layer of matrix network implemented with matrices

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")

--Input MatrixLayer Class 
local MatrixInputLayer = MatrixLayer:__new()

function MatrixInputLayer.new(size)
  return MatrixInputLayer:__new{size=size}
end

function MatrixInputLayer:initialise()
  if(self.prev ~= nil) then
    error("Inout layer at level ".. self.level .." cannot have input layer")
  end
  if(self.next == nil) then
    error("Inout layer at level ".. self.level .." has to have output layer")
  end
  self.output=torch.Tensor(self.size+1):zero()
  self.output[self.output:size(1)] = 1 
end

--Input layer only forward signal to the first hiddent layer
function MatrixInputLayer:forward()
  self.output:narrow(1,1,self.size):copy(self.input)
end

--Perform backward pass through the inout layer
function MatrixInputLayer:backwards()
  --NOOP  
end

--don't have to adjust weight in the input layer
function MatrixInputLayer:adjustWeights()
  self.next:adjustWeights()  
end

--string representation of this layer
function MatrixInputLayer.__tostring (self)
  return string.format("Input MatrixLayer of size: %d, output: %s",self.size,self.output)
end


return MatrixInputLayer