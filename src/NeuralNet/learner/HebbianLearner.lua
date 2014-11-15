--type of learner that stops after fixed number of epochs

local Learner=require("NeuralNet.learner.Learner")
local t=require("torch")

local HebbianLearner = Learner:__new()

function HebbianLearner:new(params)
  return HebbianLearner:__new(params)
end

--learn the sequence of layers using simple hebbian update rule
function HebbianLearner:learn(seq,inputSignal)  
  seq:learner(self)  
  local dataSize=inputSignal:size(1)
  for e = 1,self.nEpochs do
    local rowPermutation = torch.randperm(dataSize)
    for i = 1,dataSize do
      self.epoch = e
      local row=rowPermutation[i]
      local input=inputSignal[{row,{}}]
      self:forward(seq,input)  
      seq:adjustWeights(e)                 
    end
  end
end

--Perform forward pass through the network
function HebbianLearner:forward(seq,signal)
  seq.layers[1].input=signal
  seq:forward(function(layer)
    layer:forward()   
  end) 
end

return HebbianLearner