--Layer containing Hebbian neuron

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local TorchUtils=require("NeuralNet.utils.TorchUtils")
local t=require("torch")

--Hidden MatrixLayer class
local MatrixAggregateLayer = MatrixLayer:__new()

function MatrixAggregateLayer.new(size)
  return MatrixAggregateLayer:__new{size=size}
end

function MatrixAggregateLayer:initialise()
  if(self.prev == nil) then
    error("Hebbian layer at level ".. self.level .." has to have input layer")
  end
  self.weights=self.weightGenFun(self.size,self.prev.size)
  self.input=torch.Tensor(self.prev.size):zero()
end

--Hidden layer computes its preactivation which is tranfered through activation function which produces output sent to the next layer
function MatrixAggregateLayer:forward()
  self.output = self.weights*self.input
end

--Set weight generator
function MatrixAggregateLayer:weightGenFun(weightGenFun)
  self.weightGenFun = weightGenFun
  return self
end

function MatrixAggregateLayer.__tostring (self)
  return string.format("MatrixAggregateLayer of size: %d, output: %s, with weights:\n%s",self.size,self.output,self.weights)
end

return MatrixAggregateLayer
