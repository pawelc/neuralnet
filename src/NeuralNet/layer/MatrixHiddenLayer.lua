--MatrixLayer is the layer of matrix network implemented with matrices

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local t=require("torch")

--Hidden MatrixLayer class
local MatrixHiddenLayer = MatrixLayer:__new()

function MatrixHiddenLayer.new(size)
  return MatrixHiddenLayer:__new{size=size}
end

function MatrixHiddenLayer:initialise()
  if(self.prev == nil) then
    error("Hidden layer at level ".. self.level .." has to have input layer")
  end
  if(self.next == nil) then
    error("Hidden layer at level ".. self.level .." has to have output layer")
  end
  self.weights=self.weightGenFun(self.size,self.prev.size+1)
  self.deltaWeights=t.Tensor(self.size,self.prev.size+1):zero()
  
  self.input=torch.Tensor(self.prev.size+1):zero()
  self.input[self.input:size(1)] = 1
end

--Hidden layer computes its preactivation which is tranfered through activation function which produces output sent to the next layer
function MatrixHiddenLayer:forward()
  self.preactivation = self.weights*self.input
  self.output = self.actFun.fun(self.preactivation)
end

--Perform backward pass through the hidden layer
function MatrixHiddenLayer:backwards()
  --because output had attached 1 at the end have to remove the last element
  --alse the weight in the next layer contain bias in the last column so aso have to narrow to this region of the matrix
  self.localGradient=t.cmul(self.actFun.funD(self.output), self.next.weights:narrow(2,1,self.next.weights:size(2)-1):transpose(1,2)*self.next.localGradient)
  self.minusGradientErrorWrtWeight=torch.Tensor(self.localGradient:size(1), self.input:size(1)):zero():addr(self.localGradient,self.input)
  self.prev:backwards()
end

--adjusting weights
function MatrixHiddenLayer:adjustWeights(epoch)   
  self.deltaWeights = self.deltaWeights * self.learner.momentumFun() + self.minusGradientErrorWrtWeight*self.learner.learningRateFun()
  self.weights = self.weights + self.deltaWeights
  self.next:adjustWeights()  
end

--Set activation function
function MatrixHiddenLayer:actFun(actFun)
  self.actFun = actFun
  return self
end

--Set weight generator
function MatrixHiddenLayer:weightGenFun(weightGenFun)
  self.weightGenFun = weightGenFun
  return self
end

function MatrixHiddenLayer.__tostring (self)
  return string.format("Hidden MatrixLayer of size: %d, output: %s, localGradient: %s with weights:\n%s\ndelta weights:\n%s",self.size,self.output,self.localGradient,self.weights,self.deltaWeights)
end

return MatrixHiddenLayer
