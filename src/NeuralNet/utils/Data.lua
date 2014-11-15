--Used to process data

require 'torch'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local Learner = require 'NeuralNet.learner.Learner'

local Data={}

--read data from the file and returns Tensor
-- file - file to read from
-- nColumns - number of column in the file
-- sep - separator
-- mapping which defines how to mul n-th column of non numeric data into numeric one
function Data.fileToTensor(params)
  local file = params.file
  local nColumns = params.nColumns
  local sep = params.sep
  local mapping = params.mapping
  
  local lines=io.lines(file)
  local table = {}
  local pattern = "%s*(%S+)%s*"..sep.."%s*"..string.rep("(%S+)%s*"..sep.."%s*",nColumns-2).."(%S+).*"
  local rowNum = 0
  for l in  lines do
       rowNum = rowNum + 1
       local row={string.match(l, pattern)}
       if #row == nColumns then
        table[#table+1] = row
       else
        logger:warn(string.format("No including %d row from file: %s because parsed columns: %d",rowNum,file,#row))
       end
  end
  
  if mapping then
    for col,mappingFun in pairs(params.mapping) do
      for row = 1,#table do
        table[row][col]=mappingFun(table[row][col])
      end 
    end
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