--Trains neural net for the XOR prblem

package.path=package.path..";/Users/pawelc/git/neuralnet/src/?.lua"

local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local hidden = require 'NeuralNet.layer.MatrixHiddenLayer'
local output = require 'NeuralNet.layer.MatrixOutputLayer'
local act = require 'NeuralNet.Activation'
local err = require 'NeuralNet.Error'
local weighGen = require 'NeuralNet.WeightGen'
local learner = require 'NeuralNet.learner.Learner'
local data = require 'NeuralNet.data.data'
local t = require 'torch'

local function buildModel(neuronLayer1,neuronsLayers2)
  --create sequence of layers
  --setting seed so the experiment can be repeted 
  t.manualSeed(123)    
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(2))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(hidden.new(neuronLayer1):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero))
  seq:addLayer(hidden.new(neuronsLayers2):
                    actFun(act.tanhAct):
                    weightGenFun(weighGen.uniformAroundZero))                                   
  --add output layer with 1 output neuron with tanh activation functioin                
  seq:addLayer(output.new(1):
                  actFun(act.linAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero))
  --training without momentum                  
  seq:momemtumConstant(0)
  --learning rate
  seq:learningRate(0.1)   
  --initialize neural net             
  seq:initialise() 
  return seq   
end


local function main()
  local ndp = data.fileToTensor("/Users/pawelc/git/neuralnet/src/NeuralNet/exercises/module3/NDP.dat",3,"%s*") 
  
  local dataSetup = data.setupTrainValidationTestData(ndp,10)
  
  
  
  --define grid search for hyperparameters
  local layer1Sizes={2,3,4,5}
  local layer2Sizes={2,3,4,5}
  local nEpochs = 50
  local nFolds = 10
  local bestModel=nil
  --look for the best set of hyper parameters
  for _,l1s in ipairs(layer1Sizes) do
    for _,l2s in ipairs(layer2Sizes) do
      local rmse=data.crossValidate(buildModel(l1s,l2s),dataSetup.trainAndValidationDataSetup, nFolds,nEpochs)
      print(string.format("For layer1 size: %d, layer2 size: %d average validation RMSE is %f",l1s,l2s,rmse))
      if bestModel == nil or bestModel.rmse > rmse then
        bestModel = {rmse=rmse,l1s=l1s,l2s=l2s}
      end  
    end
  end 
  
  print(string.format("Selected model with layer 1 size: %d, layer 2 size: %d with validation RMSE error: %f",bestModel.l1s,bestModel.l2s,bestModel.rmse))    
  
  --using selected hyper parameters train model on train and validation data and compute error on test data to get genralization performance
  local seq=buildModel(bestModel.l1s,bestModel.l2s)
  learner.stopAfterNEpochsLearner(seq,nEpochs,dataSetup.trainAndValidationData[{{},{1,dataSetup.trainAndValidationData:size(2)-1}}],dataSetup.trainAndValidationData[{{},dataSetup.trainAndValidationData:size(2)}])
  local testRmse = learner.rmse(seq,dataSetup.testData[{{},{1,dataSetup.testData:size(2)-1}}],dataSetup.testData[{{},dataSetup.testData:size(2)}])
  print(string.format("Test RMSE error for the selected best model is: %f", testRmse))
end

main()
