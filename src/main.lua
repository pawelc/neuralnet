package.path=package.path..";/Users/pawelc/Projects/Lua/nuralnet/src/?.lua"

local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local hidden = require 'NeuralNet.layer.MatrixHiddenLayer'
local output = require 'NeuralNet.layer.MatrixOutputLayer'
local act = require 'NeuralNet.Activation'
local err = require 'NeuralNet.Error'
local weighGen = require 'NeuralNet.WeightGen'
local learner = require 'NeuralNet.learner.Learner'
local t = require 'torch'
require("optim")


local function main()
--  t.manualSeed(123)

  local seq=sequence:new()
  seq:addLayer(input.new(2))
  seq:addLayer(hidden.new(2):
                  actFun(act.tanhAct):
                  weightGenFun(weighGen.uniformAroundZero))                 
  seq:addLayer(output.new(1):
                  actFun(act.tanhAct):
                  errFun(err.simple):
                  weightGenFun(weighGen.uniformAroundZero))
  seq:momemtumConstant(0.3)
  seq:learningRate(0.1)                
  seq:initialise()  
  
  local inputSignal = t.Tensor({
    {-1,1},
    {1,-1},
    {-1,-1},
    {1,1}
  })
  local targetSignal = t.Tensor({
  1,
  1,
  -1,
  -1
  })
  
--  print(string.format("Before learning RMSE: %f",learner.rmse(seq,inputSignal,targetSignal)))
--  print(seq:forward(t.Tensor({-1,1}),t.Tensor({1})))
--  print(seq:forward(t.Tensor({1,-1}),t.Tensor({1})))
--  print(seq:forward(t.Tensor({-1,-1}),t.Tensor({-1})))
--  print(seq:forward(t.Tensor({1,1}),t.Tensor({-1})))
  
  
  
  learner.stopAfterNEpochsLearner(seq,1000,inputSignal,targetSignal)
  
  print(learner.rmse(seq,inputSignal,targetSignal))  
  print(learner.confusion(seq,inputSignal,targetSignal))
--  print(seq:forward(t.Tensor({-1,1}),t.Tensor({1})))
--  print(seq:forward(t.Tensor({1,-1}),t.Tensor({1})))
--  print(seq:forward(t.Tensor({-1,-1}),t.Tensor({-1})))
--  print(seq:forward(t.Tensor({1,1}),t.Tensor({-1})))
  
end

main()
