--Exercise 1. Implement the two-inputs SOM with 1D output lattice and test
--it on a square grid with random samples.

--setup project location
require 'paths'
local projectLocation = paths.dirname(paths.thisfile()).."/../../../../"
--setup path so lua can find required modules
package.path=package.path..";"..projectLocation.."/src/?.lua"

require 'pl'
local t = require 'torch'
local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local MatrixAggregateLayer = require 'NeuralNet.layer.MatrixAggregateLayer'
local UnsupervisedLearner = require 'NeuralNet.learner.UnsupervisedLearner'
local Learner = require 'NeuralNet.learner.Learner'
local AdjustParamsFunctions = require 'NeuralNet.learner.AdjustParamsFunctions'
local Data = require 'NeuralNet.utils.Data'
require "logging"
local weighGen = require 'NeuralNet.WeightGen'
local Error = require 'NeuralNet.Error'
local TableUtils = require 'NeuralNet.utils.TableUtils'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -i,--normalize_input                 should the input be normalized
   -w,--normalize_weights               should the weights be normalized after each weight update
   -e,--epochs            (default 1)   number of epochs
   -l,--learning_rate     (default 0.1) learning rate
]]

logger = logging.new(function(self, level, message)
--                             print(level, " "..message)
                             return true
                           end)

                           
--function that build model witch can be prametrized with different hyperparameters
local function buildModel()
  --create sequence of layers
  --setting seed so the experiment can be repeted 
  t.manualSeed(1243)

  --create sequence of layers
  local seq=sequence:new()
  --add layer with 4 inputs as the dimension of the input of the iris data set
  seq:addLayer(input.new(4))
  --output is the hebbian layer with the same number of neurons 
  seq:addLayer(MatrixAggregateLayer.new(1):                  
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createHebbianAdjustParamsFun(opt.normalize_weights,opt.learning_rate))
                  )                 
  
  --initialize neural net             
  seq:initialise()
  
  return seq 
end                           
  
local function main()         
  --read iris data when plant names are translated into numerical values                           
  local data = Data.stdinToTensor{nColumns=2,sep=","}
                              
  if opt.normalize_input then
    Data.normaliseData(data,{1,2})                                  
  end   
 
  local model = buildModel()
  local learner = UnsupervisedLearner:new{nEpochs=opt.epochs}:learn(model,data)
  
  for i = 1,data:size(1) do
    print(model:forwardSig(data[{i,{}}]):select(1,1))
  end    
             
end
  
main()                    