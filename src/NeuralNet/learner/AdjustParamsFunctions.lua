local AdjustParamsFunctions={}

require 'torch'
local TorchUtils = require 'NeuralNet.utils.TorchUtils'

function AdjustParamsFunctions.createHebbianAdjustParamsFun(normalizeWeights,learningRate)
  return function(layer,epoch)
    layer.deltaWeights = torch.Tensor(layer.output:size(1), layer.input:size(1)):zero():addr(layer.output,layer.input) * learningRate
    layer.weights = layer.weights + layer.deltaWeights
    --normalize weights
    if normalizeWeights then
      layer.weights = torch.div(layer.weights,math.sqrt(torch.dot(layer.weights,layer.weights)))
    end
  end
end

function AdjustParamsFunctions.createSejnowskiCovarianceRuleAdjustParamsFun(learningRate)
  local currentEpoch = 0
  return function(layer,epoch)
    if epoch ~= currentEpoch then
      currentEpoch = epoch
      layer.inputRunningSum=torch.Tensor(layer.input:size(1)):zero()
      layer.outputRunningSum=torch.Tensor(layer.output:size(1)):zero()
      layer.examplesProcessed=0  
    end
    layer.inputRunningSum = layer.inputRunningSum + layer.input
    layer.outputRunningSum = layer.outputRunningSum + layer.output
    layer.examplesProcessed=layer.examplesProcessed+1
    
    layer.inputMean=torch.div(layer.inputRunningSum,layer.examplesProcessed)
    layer.outputMean=torch.div(layer.outputRunningSum,layer.examplesProcessed)
  
    layer.deltaWeights = torch.Tensor(layer.output:size(1), layer.input:size(1)):zero():addr(layer.output-layer.outputMean,layer.input-layer.inputMean)*learningRate
    layer.weights = layer.weights + layer.deltaWeights
  end
end

function AdjustParamsFunctions.createOjaAdjustParamsFun(learningRate)
  return function(layer,epoch)
    layer.deltaWeights = torch.Tensor(layer.output:size(1), layer.input:size(1)):zero():addr(layer.output,layer.input-layer.weights*layer.output:select(1,1))*learningRate
    layer.weights = layer.weights + layer.deltaWeights
    --normalize weights
    layer.weights = torch.div(layer.weights,math.sqrt(torch.dot(layer.weights,layer.weights)))
  end
end

function AdjustParamsFunctions.createCompetitiveLearningFun(learningRate)
  return function(layer,epoch)
    local maxNeuronIdx = TorchUtils.argmax(layer.output)
    local weights = layer.weights[{{maxNeuronIdx},{}}]
    --update weights of the winning neuron
    weights = weights+(layer.input-weights)*learningRate
    --normalize
    weights = torch.div(weights,math.sqrt(torch.dot(weights,weights)))
    layer.weights[{{maxNeuronIdx},{}}] = weights
  end
end

--adjusting weights
function AdjustParamsFunctions.createHiddenLayerBackPropAdjustParamsFun(params)
  local momentum = params.momentum
  local learningRateFun = params.learningRateFun  
   
  return function(layer,epoch)
    layer.deltaWeights = layer.deltaWeights * momentum + layer.minusGradientErrorWrtWeight*learningRateFun(epoch)
    layer.weights = layer.weights + layer.deltaWeights      
  end
end


function AdjustParamsFunctions.createOutputLayerBackPropAdjustParamsFun(params)
  local momentum = params.momentum
  local learningRateFun = params.learningRateFun
  --adjusting weights
  return function(layer,epoch)  
    layer.deltaWeights = layer.deltaWeights * momentum + layer.minusGradientErrorWrtWeight * learningRateFun(epoch)
    layer.weights = layer.weights + layer.deltaWeights
  end
end

--create function computing learning rate that changes exponentially for each epoch
function AdjustParamsFunctions.createExpLearningRateFun(params)
  local initialLearningRate = params.initialLearningRate
  local expDecayConst = params.expDecayConst
  return function(epoch)
    return initialLearningRate*torch.exp(expDecayConst*(epoch-1))
  end  
end

--create function computing learning rate the is constant for each epoch
function AdjustParamsFunctions.createConstLearningRate(learningRate)
  return function(epoch)
    return learningRate
  end  
end

return AdjustParamsFunctions