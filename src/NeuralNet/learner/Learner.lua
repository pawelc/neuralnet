require("torch")

local Learner={
  shouldCheckGradient=false,
  --eps used to computer gradient numerically
  checkGradientEps=1e-4,
  --used to compare back prop gradient and numerically computed gradient
  compareGradientEps=1e-3
}

--MatrixLayer is the layer of matrix network implemented with matrices

--data should be in 3-D tensor when 1st dim is example index, 2nd dim has two choices: 
--1 for data and 2 for expected signal, the 3rd dim is the data belonging to input or expected output
function Learner.stopAfterNEpochsLearner(seq,nEpochs,inputSignal,targetSignal)  
  local dataSize=inputSignal:size(1)
  for e = 1,nEpochs do
    local rowPermutation = torch.randperm(dataSize)
    for i = 1,dataSize do
      local row=rowPermutation[i]
      local input=inputSignal[{row,{}}]
      local target=targetSignal[{{row}}]
      seq:forward(input,target)  
      seq:backwards()
      Learner.checkGradient(seq,input,target) 
      seq:adjustWeights()                 
    end
--    print(rmse(seq,inputSignal,targetSignal)) 
  end
end

--Checks error gradient wrt weights numerically if it was properly computed by back prop.
function Learner.checkGradient(seq,input,target)  
  if(Learner.shouldCheckGradient) then      
      local originalWeights={}
      --first copy original weight
      for i,l in ipairs(seq.layers) do
        if(l.weights) then
          originalWeights[i]=l.weights:clone()
        end
      end
      
      --for each weight perturb it and compute error gradient wrt it
      for i,l in ipairs(seq.layers) do
        if(l.weights) then      
          for el=0,l.weights:nElement()-1 do
            local org=l.weights:storage()[l.weights:storageOffset()+el]
            
            --perturb minus epsilon            
            l.weights:storage()[l.weights:storageOffset()+el]=org-Learner.checkGradientEps
            --have to narrow back input which is now extended
            seq:forward(input,target)
            local e1=seq:error():pow(2)*0.5
            
            --perturb plus epsilon            
            l.weights:storage()[l.weights:storageOffset()+el]=org+Learner.checkGradientEps
            seq:forward(input,target)
            local e2=seq:error():pow(2)*0.5
            
            --revert change
            l.weights:storage()[l.weights:storageOffset()+el]=org
            
            local grad=(e2-e1)/(2*Learner.checkGradientEps)
            
            if torch.abs(grad[1]+l.minusGradientErrorWrtWeight:storage()[l.minusGradientErrorWrtWeight:storageOffset()+el])>Learner.compareGradientEps then
              error(string.format("Layer: %d, weight: %d, numeric gradient: %f, back prop grad: %f",i, el,grad[1],-l.minusGradientErrorWrtWeight:storage()[l.minusGradientErrorWrtWeight:storageOffset()+el]))
            end
          end            
        end              
      end      
      
      --restore orginal weights and state
      for i,l in ipairs(seq.layers) do
        if(l.weights) then         
          l.weights=originalWeights[i]
        end
      end
      seq:forward(input,target)
  end
end

function Learner.rmse(seq,inputSignal,targetSignal)
  local dataSize=inputSignal:size(1)  
  local rmsev = 0
  for i = 1,dataSize do
    seq:forward(inputSignal[{i,{}}],targetSignal[{{i}}])  
    rmsev = seq:error()*seq:error()+rmsev                 
  end
  return torch.sqrt(rmsev/dataSize)
end

return Learner
