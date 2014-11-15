--Layer containing Hebbian neuron

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
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
end

--Hidden layer computes its preactivation which is tranfered through activation function which produces output sent to the next layer
function MatrixHebbianLayer:forward()
  self.input = self.prev.output
  self.output = self.weights*self.input
end

--adjusting weights
function MatrixHebbianLayer:adjustWeights(epoch)   
    self.weights = self.weights + self.learner.learningRateFun(epoch) * self.output * self.input  
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
