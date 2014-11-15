--Exercise 1. Implement simple Hebbian rule to perform clustering of the
--standard Iris Flower input data. Try with 3 clusters.

--setup project location
projectLocation = "/Users/pawelc/git/neuralnet"
--setup path so lua can find required modules
package.path=package.path..";/Users/pawelc/git/neuralnet/src/?.lua"

local t = require 'torch'
local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local MatrixHebbianLayer = require 'NeuralNet.layer.MatrixHebbianLayer'
local HebbianLearner = require 'NeuralNet.learner.HebbianLearner'
local Data = require 'NeuralNet.utils.Data'
require "logging"
local weighGen = require 'NeuralNet.WeightGen'

logger = logging.new(function(self, level, message)
                             print(level, " "..message)
                             return true
                           end)
  
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


--setting seed so the experiment can be repeted
  t.manualSeed(123)

  --create sequence of layers
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(4))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(MatrixHebbianLayer.new(3):                  
                  weightGenFun(weighGen.uniformAroundZero))                 
  
  --initialize neural net             
  seq:initialise()
  
  --stop after n epochs
  local nEpochs=1
  logger:info(string.format("Running HebbianLearner %d epochs",nEpochs))
  
  HebbianLearner:new{
    nEpochs=nEpochs,
    }:constLearningRate(0.1)      
      :learn(seq,iris[{{},{1,4}}])
             
  end
  
  
  main()                    