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

--Create the input MatrixLayer of given size, only invoked by concreate layer type
function Learner:setNEpochs (e)
  self.nEpochs = e
  return self
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

function Learner:rmse(seq,inputSignal,targetSignal)
  local dataSize=inputSignal:size(1)  
  local rmsev = 0
  for i = 1,dataSize do
    seq:forwardSig(inputSignal[{i,{}}],targetSignal[{{i}}])  
    rmsev = seq:error()*seq:error()+rmsev                 
  end
  return torch.sqrt(rmsev/dataSize)
end

function Learner:classError(seq,inputSignal,targetSignal)
  local dataSize=inputSignal:size(1)  
  local ce = 0
  for i = 1,dataSize do
    seq:forwardSig(inputSignal[{i,{}}],targetSignal[{i}])  
    ce = seq:error()+ce                 
  end
  return ce/dataSize
end

--create function computing learning rate that changes exponentially for each epoch
function Learner:expLearningRate(initialLearningRate,expDecayConst)
  self.learningRateFun=function()
    return initialLearningRate*torch.exp(expDecayConst*(self.epoch-1))
  end  
  return self
end

--create function computing learning rate the is constant for each epoch
function Learner:constLearningRate(learningRate)
  self.learningRateFun = function()
    return learningRate
  end  
  return self
end

--create function computing momentum rate the is constant for each epoch
function Learner:constMomentum(momentum)
  self.momentumFun = function()
    return momentum
  end  
  return self
end

--performs training and compute validation error over folds cross validation for different hyperparameters
function Learner:crossValidate(model,trainAndValidationDataSetup,folds,learner)
  local avgError = 0
  for _,trainValidData in ipairs(trainAndValidationDataSetup) do 
    learner:learn(model,trainValidData.trainInput[{{},{}}],trainValidData.trainTarget[{{},{}}])
    avgError = avgError + self:error(model,trainValidData.validInput[{{},{}}],trainValidData.validTarget[{{},{}}])
  end
  avgError=avgError/folds  
  return avgError
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
  
  --recursively creating grid of all parameters, doing exhaustive search over hyper parameters
  function hyperGridSearch(flattened,params,idx)
    if(idx <= #paramsNames) then
      local paramList = params[paramsNames[idx]]    
      for _,v in pairs(paramList) do
        flattened[paramsNames[idx]]=v
        hyperGridSearch(flattened,params,idx+1) 
      end
    else
      local error=learner:crossValidate(buildModelFun(flattened,learner),dataSetup.trainAndValidationDataSetup, nFolds,learner)
      logger:info(string.format("For params: %s\naverage validation error is %f",TableUtils.tostring(flattened),error))
      if bestModel == nil or bestModel.perf.validError > error then
        bestModel = {params=TableUtils.shallowCopy(flattened)}
        bestModel.perf = {validError=error}
      end  
     
    end
  end  
  hyperGridSearch({},params,1)  
  
  --using selected hyper parameters train model on train and validation data and compute error on test data to get genralization performance
  local seq=buildModelFun(bestModel.params,learner)
  learner:learn(seq,dataSetup.trainAndValidationInputData[{{},{}}],dataSetup.trainAndValidationTargetData[{{},{}}])
  bestModel.perf.testError = learner:error(seq,dataSetup.testInputData[{{},{}}],dataSetup.testTargetData[{{},{}}])  
  return bestModel
end

return Learner
