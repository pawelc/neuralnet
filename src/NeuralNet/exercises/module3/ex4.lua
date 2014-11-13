--Trains neural net for the NDP data

--setup project location
projectLocation = "/Users/pawelc/git/neuralnet"

package.path=package.path..";"..projectLocation.."/src/?.lua"

--import all required dependencies
local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local hidden = require 'NeuralNet.layer.MatrixHiddenLayer'
local output = require 'NeuralNet.layer.MatrixOutputLayer'
local act = require 'NeuralNet.Activation'
local err = require 'NeuralNet.Error'
local weighGen = require 'NeuralNet.WeightGen'
local StopAfterNEpochsLearner = require 'NeuralNet.learner.StopAfterNEpochsLearner'
local Learner = require 'NeuralNet.learner.Learner'
local data = require 'NeuralNet.utils.Data'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local t = require 'torch'
require "logging"

--logging
logger = logging.new(function(self, level, message)
                             print(level, message)
                             return true
                           end)
logger:setLevel (logging.INFO)

--function that build model witch can be prametrized with different hyperparameters
local function buildModel(params)
  --create sequence of layers
  --setting seed so the experiment can be repeted 
  t.manualSeed(123)    
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(2))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(hidden.new(params.layer1Size):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero))
  seq:addLayer(hidden.new(params.layer2Size):
                    actFun(act.tanhAct):
                    weightGenFun(weighGen.uniformAroundZero))                                   
  --add output layer with 1 output neuron with tanh activation functioin                
  seq:addLayer(output.new(1):
                  actFun(act.linAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero))
  --initialize neural net             
  seq:initialise() 
  return seq   
end


local function main()
  --Load the data file
  local ndp = data.fileToTensor(projectLocation.."/src/NeuralNet/exercises/module3/NDP.dat",3,"%s*") 
  
  --split data for cross validation and test data
  local dataSetup = data.setupTrainValidationTestData(ndp,10)
   
  --look for the best model using grid search over hyper parameters
  local bestModel = Learner.hyperGridSearch{
    params = {layer1Size={1,2},layer2Size={1,2}},
    buildModelFun = buildModel,
    dataSetup=dataSetup,
    nFolds=10,
    learner = StopAfterNEpochsLearner:new{nEpochs=50,shouldCheckGradient=false}
      :expLearningRate(0.1,1.1)
      :constMomentum(0)
  }
  
  --show best found model
  logger:info(string.format("Selected model with params: %s, performance: %s",TableUtils.tostring(bestModel.params),TableUtils.tostring(bestModel.perf)))
end

main()
