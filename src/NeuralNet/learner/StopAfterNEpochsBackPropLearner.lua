--type of learner that stops after fixed number of epochs

local Learner=require("NeuralNet.learner.Learner")
local t=require("torch")

local StopAfterNEpochsBackPropLearner = Learner:__new()

function StopAfterNEpochsBackPropLearner:new(params)
  return StopAfterNEpochsBackPropLearner:__new(params)
end

--data should be in 3-D tensor when 1st dim is example index, 2nd dim has two choices: 
--1 for data and 2 for expected signal, the 3rd dim is the data belonging to input or expected output
function StopAfterNEpochsBackPropLearner:learn(seq,inputSignal,targetSignal)  
  seq:learner(self)  
  local dataSize=inputSignal:size(1)
  for e = 1,self.nEpochs do
    local rowPermutation = torch.randperm(dataSize)
    for i = 1,dataSize do
      self.epoch = e
      local row=rowPermutation[i]
      local input=inputSignal[{row,{}}]
      local target=targetSignal[{{row}}]
      seq:forwardSig(input,target)  
      seq:backwards()
      Learner.checkGradient(seq,input,target) 
      seq:forwardFun(function(layer)
        if layer.adjustParamsFun then
          layer.adjustParamsFun(layer, e)
        end
      end
      )
    end
  end
end

function StopAfterNEpochsBackPropLearner:error(seq,inputSignal,targetSignal)
  return self:rmse(seq,inputSignal,targetSignal)
end

return StopAfterNEpochsBackPropLearner