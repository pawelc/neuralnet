--Exercise 1. Implement simple Hebbian rule to perform clustering of the
--standard Iris Flower input data. Try with 3 clusters.

--setup project location
projectLocation = "/Users/pawelc/git/neuralnet"
--setup path so lua can find required modules
package.path=package.path..";/Users/pawelc/git/neuralnet/src/?.lua"

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
  local iris = Data.fileToTensor{file=projectLocation.."/src/NeuralNet/exercises/module4/iris.data.txt",
                               nColumns=5,
                               sep=",",
                               mapping={[5]=function(el) 
                                  if el == "Iris-setosa" then return 1                                   
                                  elseif el == "Iris-versicolor" then return 2
                                  elseif el == "Iris-virginica" then return 3
                                end  
                               end}
                              }
  if opt.normalize_input then                              
    --normalizing input data                              
    for i = 1,4 do
      iris[{{},i}]:add(-iris[{{},i}]:mean())
      iris[{{},i}]:div(iris[{{},i}]:std())
    end
  end   
  
  --input data is only first 4 columns
  local inputData = iris[{{},{1,4}}] 
 
  local model = buildModel()
  local learner = UnsupervisedLearner:new{nEpochs=opt.epochs}:learn(model,inputData)
  
  for i = 1,iris:size(1) do
    print(model:forwardSig(inputData[{i,{}}]):select(1,1)..","..iris[i][5])
  end    
             
end
  
main()                    