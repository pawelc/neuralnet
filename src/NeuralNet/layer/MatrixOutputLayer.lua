--MatrixLayer is the layer of matrix network implemented with matrices

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local t=require("torch")

--Output MatrixLayer class
local MatrixOutputLayer = MatrixLayer:__new()


function MatrixOutputLayer.new(size)
  return MatrixOutputLayer:__new{size=size}
end

function MatrixOutputLayer:initialise()
  if(self.prev == nil) then
    error("Output layer at level ".. self.level .." has to have input layer")
  end
  if(self.next ~= nil) then
    error("Output layer at level ".. self.level .." cannot have output layer")
  end
  self.weights=self.weightGenFun(self.size,self.prev.size+1)
  self.deltaWeights=t.Tensor(self.size,self.prev.size+1):zero()
  
  self.input=torch.Tensor(self.prev.size+1):zero()
  self.input[self.input:size(1)] = 1
end

--Output layer computes its preactivation which is tranfered through activation function which produces output
function MatrixOutputLayer:forward()
  self.input:narrow(1,1,self.prev.size):copy(self.prev.output)
  self.preactivation = self.weights*self.input
  self.output = self.actFun.fun(self.preactivation)
  self.error = self.errFun(self.expected,self.output)
end

--Perform backward pass through the output layer
function MatrixOutputLayer:backwards()
  self.localGradient=t.cmul(self.error,self.actFun.funD(self.output))
  self.minusGradientErrorWrtWeight=torch.Tensor(self.size, self.input:size(1)):zero():addr(self.localGradient,self.input)
  self.prev:backwards()
end

--adjusting weights
function MatrixOutputLayer:adjustWeights(epoch)  
  self.deltaWeights = self.deltaWeights * self.learner.momentumFun() + self.minusGradientErrorWrtWeight * self.learner.learningRateFun()
  self.weights = self.weights + self.deltaWeights
end

--Set activation function
function MatrixOutputLayer:actFun(actFun)
  self.actFun = actFun
  return self
end

--Set error function
function MatrixOutputLayer:errFun(errFun)
  self.errFun = errFun
  return self
end

--Set weight generator
function MatrixOutputLayer:weightGenFun(weightGenFun)
  self.weightGenFun = weightGenFun
  return self
end

function MatrixOutputLayer.__tostring (self)
  return string.format("Output MatrixLayer of size %d, output: %s, error %s, localGradient: %s with weights:\n%s\ndelta weights:\n%s",self.size,self.output,self.error,self.localGradient,self.weights,self.deltaWeights)
end

return MatrixOutputLayer