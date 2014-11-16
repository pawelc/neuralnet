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
local StopAfterNEpochsBackPropLearner = require 'NeuralNet.learner.StopAfterNEpochsBackPropLearner'
local Learner = require 'NeuralNet.learner.Learner'
local data = require 'NeuralNet.utils.Data'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local t = require 'torch'
require "logging"
local AdjustParamsFunctions = require 'NeuralNet.learner.AdjustParamsFunctions'

--logging
logger = logging.new(function(self, level, message)
                             print(level, message)
                             return true
                           end)
logger:setLevel (logging.INFO)

--function that build model witch can be prametrized with different hyperparameters
local function buildModel(params,learner)
  --create sequence of layers
  --setting seed so the experiment can be repeted 
  t.manualSeed(123)    
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(2))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(hidden.new(params.layer1Size):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createHiddenLayerBackPropAdjustParamsFun{momentum=0,learningRateFun=AdjustParamsFunctions.
                                                                                                    createConstLearningRate(0.1)}))
  seq:addLayer(hidden.new(params.layer2Size):
                    actFun(act.tanhAct):
                    weightGenFun(weighGen.uniformAroundZero):
                    setAdjustParamsFun(AdjustParamsFunctions.createHiddenLayerBackPropAdjustParamsFun{momentum=0,learningRateFun=AdjustParamsFunctions.
                                                                                                    createConstLearningRate(0.1)}))                               
  --add output layer with 1 output neuron with tanh activation functioin                
  seq:addLayer(output.new(1):
                  actFun(act.linAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createOutputLayerBackPropAdjustParamsFun{momentum=0,learningRateFun=AdjustParamsFunctions.
                                                                                                    createConstLearningRate(0.1)}))  
  --initialize neural net             
  seq:initialise() 
  return seq   
end


local function main()
  --Load the data file
  local ndp = data.fileToTensor{file=projectLocation.."/src/NeuralNet/exercises/module3/NDP.dat",
                               nColumns=3,
                               sep="%s"}
  
  --split data for cross validation and test data
  local dataSetup = data.setupTrainValidationTestData(ndp[{{},{1,2}}],ndp[{{},{3}}],10)
   
  --look for the best model using grid search over hyper parameters
  local bestModel = Learner.hyperGridSearch{
--    params = {layer1Size={1,2,3,4,5,6,7,8,9,10},layer2Size={1,2,3,4,5,6,7,8,9,10}},
    params = {layer1Size={1,2},layer2Size={1}},
    buildModelFun = buildModel,
    dataSetup=dataSetup,
    nFolds=10,
    learner=StopAfterNEpochsBackPropLearner:new{nEpochs=50,shouldCheckGradient=false}
    }
  
  --show best found model
  logger:info(string.format("Selected model with params: %s,\nperformance: %s",TableUtils.tostring(bestModel.params),TableUtils.tostring(bestModel.perf)))
end

main()
