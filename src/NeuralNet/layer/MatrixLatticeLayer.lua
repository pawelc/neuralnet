--Layer containing neurons which compute output as dot product of its weight and input

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local TorchUtils=require("NeuralNet.utils.TorchUtils")
local t=require("torch")

--Hidden MatrixLayer class
local MatrixLatticeLayer = MatrixLayer:__new()

function MatrixLatticeLayer.new(dims)
  return MatrixLatticeLayer:__new{dims=dims}
end

function MatrixLatticeLayer:initialise()
  if(self.prev == nil) then
    error("MatrixLatticeLayer layer at level ".. self.level .." has to have input layer")
  end
  self.weights=self.weightGenFun(self.dimensions,self.prev.size)
  self.input=torch.Tensor(self.prev.size):zero()
end

function MatrixLatticeLayer:forward()
  self.output = self.weights*self.input
end

--Set weight generator
function MatrixLatticeLayer:weightGenFun(weightGenFun)
  self.weightGenFun = weightGenFun
  return self
end

function MatrixLatticeLayer.__tostring (self)
  return string.format("MatrixLatticeLayer of dims: %s, output: %s, with weights:\n%s",self.dims,self.output,self.weights)
end

return MatrixLatticeLayer
