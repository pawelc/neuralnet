local AdjustParamsFunctions={}

require 'torch'

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