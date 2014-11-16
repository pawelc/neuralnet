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
local Learner = require 'NeuralNet.learner.Learner'
local Data = require 'NeuralNet.utils.Data'
require "logging"
local weighGen = require 'NeuralNet.WeightGen'
local Error = require 'NeuralNet.Error'
local TableUtils = require 'NeuralNet.utils.TableUtils'

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
  seq:addLayer(MatrixHebbianLayer.new(1):                  
                  weightGenFun(weighGen.uniformAroundZero):
                  errFun(Error.classification))                 
  
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
  --normalizing input data                              
  for i = 1,4 do
     iris[{{},i}]:add(-iris[{{},i}]:mean())
     iris[{{},i}]:div(iris[{{},i}]:std())
  end   
  
  --input data is only first 4 columns
  local inputData = iris[{{},{1,4}}] 
 
  local model = buildModel()
  local learner = HebbianLearner:new{nEpochs=1000}:
    constLearningRate(0.1):learn(model,inputData)
  
  for i = 1,iris:size(1) do
    print(model:forwardSig(inputData[{i,{}}]):select(1,1)..","..iris[i][5])
  end    
             
end
  
  
main()                    