--Layer containing Hebbian neuron

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local TorchUtils=require("NeuralNet.utils.TorchUtils")
local t=require("torch")

--Hidden MatrixLayer class
local MatrixHebbianLayer = MatrixLayer:__new()

function MatrixHebbianLayer.new(size)
  return MatrixHebbianLayer:__new{size=size}
end

function MatrixHebbianLayer:initialise()
  if(self.prev == nil) then
    error("Hebbian layer at level ".. self.level .." has to have input layer")
  end
  self.weights=self.weightGenFun(self.size,self.prev.size)
  self.input=torch.Tensor(self.prev.size):zero()
end

--Hidden layer computes its preactivation which is tranfered through activation function which produces output sent to the next layer
function MatrixHebbianLayer:forward()
  self.output = self.weights*self.input
end

--adjusting weights
function MatrixHebbianLayer:adjustWeights(epoch)   
    self.deltaWeights = torch.Tensor(self.output:size(1), self.input:size(1)):zero():addr(self.output,self.input) * self.learner.learningRateFun()
    self.weights = self.weights + self.deltaWeights
    --normalize weights
    if self.learner.normalizeWeights then
      self.weights = torch.div(self.weights,math.sqrt(torch.dot(self.weights,self.weights)))
    end 
end

--Set weight generator
function MatrixHebbianLayer:weightGenFun(weightGenFun)
  self.weightGenFun = weightGenFun
  return self
end

function MatrixHebbianLayer.__tostring (self)
  return string.format("MatrixHebbianLayer of size: %d, output: %s, with weights:\n%s",self.size,self.output,self.weights)
end

return MatrixHebbianLayer
