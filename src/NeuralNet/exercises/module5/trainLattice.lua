--require 'cutorch'
--torch.setdefaulttensortype('torch.CudaTensor')

--setup project location
require 'paths'
local projectLocation = paths.dirname(paths.thisfile()).."/../../../../"
--setup path so lua can find required modules
package.path=package.path..";"..projectLocation.."/src/?.lua"

require 'pl'
local t = require 'torch'
local sequence = require 'NeuralNet.Sequence'
local input = require 'NeuralNet.layer.MatrixInputLayer'
local MatrixLatticeLayer = require 'NeuralNet.layer.MatrixLatticeLayer'
local UnsupervisedLearner = require 'NeuralNet.learner.UnsupervisedLearner'
local Learner = require 'NeuralNet.learner.Learner'
local AdjustParamsFunctions = require 'NeuralNet.learner.AdjustParamsFunctions'
local Data = require 'NeuralNet.utils.Data'
require "logging"
local weighGen = require 'NeuralNet.WeightGen'
local Error = require 'NeuralNet.Error'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local LuaUtils = require 'NeuralNet.utils.LuaUtils'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -e,--epochs            (default 1)   number of epochs
   -l,--learning_rate     (default 0.1) learning rate
   -s,--init_sigma        (default 3)   initial sigma
   -t,--time_decay        (default 10)  time decay
   -n,--neurons           (default "5,1")dimension of neurons in the lattice
   -d,--data_dim          (default 2)   dimension of input data
   -w,--output_weights                  output weights
   -r,--tau_2             (default -1)  tau 2 when exponetially decreasing learning rate, has to be greater than 0
   -o,--topo_fun          (default "{name='gauss'}") topological neighberhood, can be gauss: {name='gauss'} or mexican hat: {name='mex_hat',beta=x,alpha=y}
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
  seq:addLayer(input.new(opt.data_dim))
  --output is the hebbian layer with the same number of neurons
  local learningRateFun=nil 
  if opt.tau_2>0 then
    learningRateFun = AdjustParamsFunctions.createExpLearningRateFun{initialLearningRate=opt.learning_rate,expDecayConst=-1/opt.tau_2}
  else
    learningRateFun = AdjustParamsFunctions.createConstLearningRate(opt.learning_rate)    
  end
  
  loadstring("topoFunDescr="..opt.topo_fun)()
  local gaussFun=function(sigma,ds) return torch.exp(-ds/(2*sigma*sigma)) end
  local topoFun = nil
  if topoFunDescr["name"]=="gauss" then
    topoFun = gaussFun
  elseif topoFunDescr["name"]=="mex_hat" then
    topoFun = function(sigma,ds) return gaussFun(sigma,ds) - gaussFun(sigma*topoFunDescr["beta"],ds)*topoFunDescr["alpha"]  end 
  else
    error("not recognized topological function descriptor: "..topoFunDescr)
  end
  
  
  seq:addLayer(MatrixLatticeLayer.new({dims=t.LongStorage(LuaUtils.csvStringToTable(opt.neurons))}):                  
                  weightGenFun(weighGen.uniformAroundZero):
                  setAdjustParamsFun(AdjustParamsFunctions.createSOMAdjustParamsFun{tnFun=topoFun,learningRateFun=learningRateFun,initSigma=opt.init_sigma,timeDecay=opt.time_decay})
                  )                 
  
  --initialize neural net             
  seq:initialise()
  
  return seq 
end      

local function produceHeader()
  local latticeDims = LuaUtils.csvStringToTable(opt.neurons)
  header={}
  for i =1,#latticeDims do
    header[#header+1] = "d"..i
  end
  
  for i =1,opt.data_dim do
    header[#header+1] = "w"..i
  end
  
  return table.concat(header,", ")
end
  
local function main()         
--  timer = torch.Timer()
  --read iris data when plant names are translated into numerical values                           
  local data = Data.stdinToTensor{nColumns=opt.data_dim,sep=","}
--  local data = Data.fileToTensor{file="src/NeuralNet/exercises/module5/input.txt",nColumns=2,sep=","}
--local data = Data.fileToTensor{file="src/NeuralNet/exercises/module5/iris.data.txt",nColumns=4,sep=","}
                                
  local model = buildModel()
  local learner = UnsupervisedLearner:new{nEpochs=opt.epochs}:learn(model,data)
  
  if opt.output_weights then
    print(produceHeader())
    print(TorchUtils.twoDimLatticeToCsv(model.layers[2].weights))
  end        
--  print('Time: ' .. timer:time().real .. ' seconds')           
end
  
main()                    