require("torch")

--MatrixLayer is the layer of matrix network implemented with matrices

--data should be in 3-D tensor when 1st dim is example index, 2nd dim has two choices: 
--1 for data and 2 for expected signal, the 3rd dim is the data belonging to input or expected output
local function stopAfterNEpochsLearner(seq,nEpochs,inputSignal,targetSignal)  
  local dataSize=inputSignal:size(1)
  for e = 1,nEpochs do
    rowPermutation = torch.randperm(dataSize)
    for i = 1,dataSize do
      row=rowPermutation[i]
      seq:forward(inputSignal[{row,{}}],targetSignal[{{row}}])  
      seq:backwards()                 
    end
    
    
--    print(rmse(seq,inputSignal,targetSignal)) 
  end
end

function rmse(seq,inputSignal,targetSignal)
  local dataSize=inputSignal:size(1)  
  local rmse = 0
  for i = 1,dataSize do
    seq:forward(inputSignal[{i,{}}],targetSignal[{{i}}])  
    rmse = rmse+seq:error()*seq:error()                 
  end
  return torch.sqrt(rmse/dataSize)
end

function confusion(seq,inputSignal,targetSignal)
  local dataSize=inputSignal:size(1)  
  classes = {'-1','1'}
  local confusion = optim.ConfusionMatrix(classes)
  for i = 1,dataSize do
    seq:forward(inputSignal[{i,{}}],targetSignal[{{i}}])
    confusion:add(seq.output, targetSignal[{{i}}])                       
  end
  return confusion
end

return {
  stopAfterNEpochsLearner=stopAfterNEpochsLearner,
  rmse=rmse,
  confusion=confusion
}