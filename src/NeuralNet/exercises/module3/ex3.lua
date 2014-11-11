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
local t = require 'torch'

local function main()
  --setting seed so the experiment can be repeted
  t.manualSeed(123)

  --create sequence of layers
  local seq=sequence:new()
  --add layer with 2 inputs
  seq:addLayer(input.new(2))
  --add hiddent layer with 2 neurons, tanh activation and uniform weight initialization 
  seq:addLayer(hidden.new(1):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero))
seq:addLayer(hidden.new(1):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero))                                   
  --add output layer with 1 output neuron with tanh activation functioin                
  seq:addLayer(output.new(1):
                  actFun(act.tanhAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero))
  --training without momentum                  
  seq:momemtumConstant(0)
  --learning rate
  seq:learningRate(0.1)   
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
  
  print(string.format("Before learning RMSE: %f",learner.rmse(seq,inputSignal,targetSignal)))
  
  --we can check gradient numerically to debug implementation
  learner.shouldCheckGradient=false
  --stop after n epochs
  local nEpochs=1000
  print(string.format("Running learner %d epochs",nEpochs))
  learner.stopAfterNEpochsLearner(seq,nEpochs,inputSignal,targetSignal)
  
  --check how we did with learning
  print(string.format("After learning RMSE: %f",learner.rmse(seq,inputSignal,targetSignal)))  
  print(string.format("Trained answer for input: -1,1 is %s",seq:forward(t.Tensor({-1,1}),t.Tensor({1}))))
  print(string.format("Trained answer for input: 1,-1 is %s",seq:forward(t.Tensor({1,-1}),t.Tensor({1}))))
  print(string.format("Trained answer for input: -1,-1 is %s",seq:forward(t.Tensor({-1,-1}),t.Tensor({-1}))))
  print(string.format("Trained answer for input: 1,1 is %s",seq:forward(t.Tensor({1,1}),t.Tensor({-1}))))
  
  print(string.format("Weights in the first hidden layer:\n%s",seq.layers[2].weights))
  print(string.format("Weights in the output layer:\n%s",seq.layers[3].weights))
  
end

main()
