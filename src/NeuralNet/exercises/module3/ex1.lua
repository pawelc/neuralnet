--Trains neural net for the XOR prblem

package.path=package.path..";/Users/pawelc/git/neuralnet/src/?.lua"

local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local hidden = require 'NeuralNet.layer.MatrixHiddenLayer'
local output = require 'NeuralNet.layer.MatrixOutputLayer'
local AdjustParamsFunctions = require 'NeuralNet.learner.AdjustParamsFunctions'
local act = require 'NeuralNet.Activation'
local err = require 'NeuralNet.Error'
local weighGen = require 'NeuralNet.WeightGen'
local StopAfterNEpochsBackPropLearner = require 'NeuralNet.learner.StopAfterNEpochsBackPropLearner'
local Learner = require 'NeuralNet.learner.Learner'
local t = require 'torch'
require "logging"

logger = logging.new(function(self, level, message)
                             print(level, " "..message)
                             return true
                           end)
logger:setLevel (logging.INFO)

local function main()
  --setting seed so the experiment can be repeted
  t.manualSeed(123)

  --create sequence of layers
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(2))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(hidden.new(2):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createHiddenLayerBackPropAdjustParamsFun{momentum=0,
                                                                                                    learningRateFun=AdjustParamsFunctions.
                                                                                                    createConstLearningRate(0.1)}))                 
  --add output layer with 1 output neuron with tanh activation functioin                
  seq:addLayer(output.new(1):
                  actFun(act.tanhAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createOutputLayerBackPropAdjustParamsFun{momentum=0,learningRateFun=AdjustParamsFunctions.
                                                                                                    createConstLearningRate(0.1)}))
  
  --initialize neural net             
  seq:initialise()  
  
  --training data
  local inputSignal = t.Tensor({
    {-1,1},
    {1,-1},
    {-1,-1},
    {1,1}
  })
  --with target labels
  local targetSignal = t.Tensor({
  1,
  1,
  -1,
  -1
  })
  
  --stop after n epochs
  local nEpochs=1000
  logger:info(string.format("Running learner %d epochs",nEpochs))
  
  local learner = StopAfterNEpochsBackPropLearner:new{
    nEpochs=nEpochs,
    shouldCheckGradient=false}
  
  logger:info(string.format("Before learning RMSE: %f",learner:rmse(seq,inputSignal,targetSignal)))
  
  learner:learn(seq,inputSignal,targetSignal)
  
  --check how we did with learning
  logger:info(string.format("After learning RMSE: %f",learner:rmse(seq,inputSignal,targetSignal)))  
  logger:info(string.format("Trained answer for input: -1,1 is %s",seq:forwardSig(t.Tensor({-1,1}),t.Tensor({1}))))
  logger:info(string.format("Trained answer for input: 1,-1 is %s",seq:forwardSig(t.Tensor({1,-1}),t.Tensor({1}))))
  logger:info(string.format("Trained answer for input: -1,-1 is %s",seq:forwardSig(t.Tensor({-1,-1}),t.Tensor({-1}))))
  logger:info(string.format("Trained answer for input: 1,1 is %s",seq:forwardSig(t.Tensor({1,1}),t.Tensor({-1}))))
  
  logger:info(string.format("Weights in the first hidden layer:\n%s",seq.layers[2].weights))
  logger:info(string.format("Weights in the output layer:\n%s",seq.layers[3].weights))
  
end

main()
