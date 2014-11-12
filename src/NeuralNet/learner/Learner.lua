require("torch")
local TableUtils = require 'NeuralNet.utils.TableUtils'

local Learner={
  shouldCheckGradient=false,
  --eps used to computer gradient numerically
  checkGradientEps=1e-4,
  --used to compare back prop gradient and numerically computed gradient
  compareGradientEps=1e-3
}

--Create the input MatrixLayer of given size, only invoked by concreate layer type
function Learner:__new (o)
  o = o or {}
  setmetatable(o, self)
  self.__index = self
  return o
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

--performs training and compute validation error over folds cross validation for different hyperparameters
function Learner.crossValidate(model,trainAndValidationDataSetup,folds,learner)
  local avgRmse = 0
  for _,trainValidData in ipairs(trainAndValidationDataSetup) do 
    learner:learn(model,trainValidData.train[{{},{1,trainValidData.train:size(2)-1}}],trainValidData.train[{{},trainValidData.train:size(2)}])
    avgRmse = avgRmse + Learner.rmse(model,trainValidData.valid[{{},{1,trainValidData.valid:size(2)-1}}],trainValidData.valid[{{},trainValidData.valid:size(2)}])
  end
  avgRmse=avgRmse/folds  
  return avgRmse
end

--performs grid search on different configuration of hyper parameters deffined.
-- params - table with list per parameter
-- buildModelFun - function buildnig model for one conbination of hyper parameters 
-- dataSetup - data split for cross validation folds and test data
function Learner.hyperGridSearch(opts)
  local params=opts.params
  local buildModelFun=opts.buildModelFun
  local dataSetup=opts.dataSetup
  local nFolds=opts.nFolds
  local nFolds=opts.nFolds
  local learner=opts.learner
  
  local paramsNames={}
  local i = 1
  for paramName,_ in pairs(params) do
    paramsNames[i]=paramName
    i=i+1
  end
  local bestModel=nil
  
  function hyperGridSearch(flattened,params,idx)
    if(idx <= #paramsNames) then
      local paramList = params[paramsNames[idx]]    
      for _,v in pairs(paramList) do
        flattened[paramsNames[idx]]=v
        hyperGridSearch(flattened,params,idx+1) 
      end
    else
      local rmse=Learner.crossValidate(buildModelFun(flattened),dataSetup.trainAndValidationDataSetup, nFolds,learner)
      logger:info(string.format("For params: %s average validation RMSE is %f",TableUtils.tostring(flattened),rmse))
      if bestModel == nil or bestModel.perf.validRmse > rmse then
        bestModel = {params=TableUtils.shallowCopy(flattened)}
        bestModel.perf = {validRmse=rmse}
      end  
     
    end
  end  
  hyperGridSearch({},params,1)  
  
  --using selected hyper parameters train model on train and validation data and compute error on test data to get genralization performance
  local seq=buildModelFun(bestModel.params)
  learner.learn(seq,dataSetup.trainAndValidationData[{{},{1,dataSetup.trainAndValidationData:size(2)-1}}],dataSetup.trainAndValidationData[{{},dataSetup.trainAndValidationData:size(2)}])
  bestModel.perf.testRmse = learner.rmse(seq,dataSetup.testData[{{},{1,dataSetup.testData:size(2)-1}}],dataSetup.testData[{{},dataSetup.testData:size(2)}])  
  return bestModel
end

return Learner
