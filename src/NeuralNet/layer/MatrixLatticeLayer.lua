--Layer containing neurons which compute output as dot product of its weight and input

local MatrixLayer=require("NeuralNet.layer.MatrixLayer")
local TorchUtils=require("NeuralNet.utils.TorchUtils")
local t=require("torch")

--Hidden MatrixLayer class
local MatrixLatticeLayer = MatrixLayer:__new()

function MatrixLatticeLayer.new(params)
  return MatrixLatticeLayer:__new{params=params}
end

function MatrixLatticeLayer:initialise()
  if(self.prev == nil) then
    error("MatrixLatticeLayer layer at level ".. self.level .." has to have input layer")
  end
  local weightDims = self.params.dims
--  self.output=torch.Tensor(weightDims)
  weightDims:resize(self.params.dims:size(1)+1,0) 
  weightDims[weightDims:size(1)]=self.prev.size
  self.weights=self.weightGenFun(weightDims)
  self.input=torch.Tensor(self.prev.size):zero()  
end

function MatrixLatticeLayer:forward()
--  self.maxOutput = nil
  self.minDistance = nil  
  for x = 1,self.weights:size(1) do
    for y = 1,self.weights:size(2) do
      local diff = self.input - self.weights[{x,y,{}}]
      local distance = diff * diff   
      if self.minDistance == nil or self.minDistance > distance then
        self.minDistance = distance
        self.minIdx = torch.LongStorage({x,y})
      end  
    end
  end
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
