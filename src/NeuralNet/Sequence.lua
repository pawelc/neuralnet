--Neural Net module
local t=require("torch")

--Neural Net Module
local Sequence = {}

--Create new Neural Net
function Sequence:new ()
  local o = {layers={}}
  setmetatable(o, self)
  self.__index = self
  return o
end
--NN as string
function Sequence:__tostring ()
  local t = {}
  for i,layer in ipairs(self.layers) do
    t[i]=string.format("Layer %d: %s",i,layer)
  end
  return table.concat(t,"\n")
end

--Add a layer to NN
function Sequence:addLayer (layer)
  self.layers[#self.layers+1]=layer
  layer.level=#self.layers
  layer.prev=self.layers[#self.layers-1]
  if(self.layers[#self.layers-1]) then
    self.layers[#self.layers-1].next = layer
  end
end

function Sequence:initialise()
  for _,layer in ipairs(self.layers) do
    layer:initialise()
  end
end

--Perform forward pass through the network
function Sequence:forward(signal,expected)
  self.layers[#self.layers].expected=expected
  self.layers[1]:forward(signal)
  return self.layers[#self.layers].output
end

--Perform backward pass through the network
function Sequence:backwards()
  --starting from the output layer
  self.layers[#self.layers]:backwards()  
end

--Perform adjustment of weights
function Sequence:adjustWeights()
  --now recompute weights of all layers
  self.layers[1]:adjustWeights()  
end

--Returns last computed result from the network after forwarding signal through it
function Sequence:output()
  return self.layers[#self.layers].output
end

--Returns error from the last propagated function signal
function Sequence:error()
  return self.layers[#self.layers].error
end

--Sets momentum constant of each layer
function Sequence:momemtumConstant(momemtumConstant)
  for _,l in ipairs(self.layers) do
    l.momemtumConstant=momemtumConstant
  end    
end

--Sets learning rate of each layer
function Sequence:learningRate(learningRate)
  for _,l in ipairs(self.layers) do
    l.learningRate=learningRate
  end    
end

return Sequence