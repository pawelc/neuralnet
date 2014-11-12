--Used to process data

require 'torch'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local Learner = require 'NeuralNet.learner.Learner'

local Data={}

function Data.fileToTensor(file,nColumns,sep)
  local lines=io.lines(file)
  local table = {}
  for l in  lines do
       table[#table+1] = {string.match(l, string.rep(sep.."(%S+)"..sep,nColumns))}
  end
  return torch.Tensor(table)
end

--splits data into test data and validation/train folds 
function Data.setupTrainValidationTestData(data,folds)
  --create permutation of index into data so we can randomly split into test data and trainAndValidationData
  local shuffle = torch.randperm(data:size(1))
  local testDataIdx = torch.floor(0.15*data:size(1))
  
  local testData=torch.Tensor() 
  
  testData:index(data,1,torch.Tensor.long(shuffle[{{1,testDataIdx}}]))
  local trainAndValidationData = data:index(1,torch.Tensor.long(shuffle[{{testDataIdx+1,data:size(1)}}]))
  
  
  local itemsInFold=torch.floor(trainAndValidationData:size(1)/folds)
  local trainAndValidationDataSetup={}
  for fold = 1,folds do 
    local validStartIdx=1+(fold-1)*itemsInFold
    local validEndIdx = fold*itemsInFold
    local validData=trainAndValidationData[{{validStartIdx,validEndIdx},{}}]
    local trainData=torch.Tensor(trainAndValidationData:size(1)-validData:size(1),data:size(2))
    if(validStartIdx > 1) then
      trainData[{{1,validStartIdx-1},{}}]=trainAndValidationData[{{1,validStartIdx-1},{}}]
    end
    if(validEndIdx+1 < trainAndValidationData:size(1)) then
      trainData[{{validStartIdx,trainData:size(1)},{}}]=trainAndValidationData[{{validEndIdx+1,trainAndValidationData:size(1)},{}}]
    end
    
    trainAndValidationDataSetup[fold]={train=trainData,valid=validData}
  end
  return {trainAndValidationDataSetup=trainAndValidationDataSetup,testData=testData,trainAndValidationData=trainAndValidationData}
end 

return Data