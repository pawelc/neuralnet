--Used to process data

require 'torch'
local TableUtils = require 'NeuralNet.utils.TableUtils'
local Learner = require 'NeuralNet.learner.Learner'

local Data={}

---Read data from the file and returns Tensor. All parameters are passed in one table.
-- @param file file to read from
-- @param nColumns number of column in the file
-- @param sep separator
-- @param mapping which defines how to mul n-th column of non numeric data into numeric one
function Data.fileToTensor(params)
  local file = params.file
  params.lines=io.lines(file)
  return Data.linesToTensor(params)
end

--read data from the stdin and returns Tensor
-- nColumns - number of column in the file
-- sep - separator
-- mapping which defines how to mul n-th column of non numeric data into numeric one
function Data.stdinToTensor(params)
  local file = params.file
  params.lines=io.lines()
  return Data.linesToTensor(params)
end

--read data from the lines and returns Tensor
-- nColumns - number of column in the file
-- sep - separator
-- mapping which defines how to mul n-th column of non numeric data into numeric one
function Data.linesToTensor(params)
  local nColumns = params.nColumns
  local sep = params.sep
  local mapping = params.mapping
  local lines = params.lines
  
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
    for col,mappingFun in pairs(mapping) do
      for row = 1,#table do
        translated=mappingFun(table[row][col])
        if type(translated) == "number" then
          table[row][col] = translated
        elseif type(translated) == "table" then
          --remove translated element
          table[row][#table[row]]=nil
          --append translated table into current row
          for _,v in ipairs(translated) do table[row][#table[row]+1] = v end
        else
          error("Cannot decode: "..translated)
        end
      end
    end
  end

  return torch.Tensor(table)
end

--splits data into test data and validation/train folds
function Data.setupTrainValidationTestData(inputData,targetData,folds)
  --create permutation of index into data so we can randomly split into test data and trainAndValidationData
  local shuffle = torch.randperm(inputData:size(1))
  local testDataIdx = torch.floor(0.15*inputData:size(1))

  local inputTestData=torch.Tensor()
  local targetTestData=torch.Tensor()

  inputTestData:index(inputData,1,torch.Tensor.long(shuffle[{{1,testDataIdx}}]))
  targetTestData:index(targetData,1,torch.Tensor.long(shuffle[{{1,testDataIdx}}]))

  local trainAndValidationInputData = inputData:index(1,torch.Tensor.long(shuffle[{{testDataIdx+1,inputData:size(1)}}]))
  local trainAndValidationTargetData = targetData:index(1,torch.Tensor.long(shuffle[{{testDataIdx+1,inputData:size(1)}}]))

  local itemsInFold=torch.floor(trainAndValidationInputData:size(1)/folds)
  local trainAndValidationDataSetup={}

  for fold = 1,folds do
    local validStartIdx=1+(fold-1)*itemsInFold
    local validEndIdx = fold*itemsInFold

    local validInputData=trainAndValidationInputData[{{validStartIdx,validEndIdx},{}}]
    local validTargetData=trainAndValidationTargetData[{{validStartIdx,validEndIdx},{}}]

    local trainInputData=torch.Tensor(trainAndValidationInputData:size(1)-validInputData:size(1),inputData:size(2))
    local trainTargetData=torch.Tensor(trainAndValidationTargetData:size(1)-validTargetData:size(1),targetData:size(2))

    if(validStartIdx > 1) then
      trainInputData[{{1,validStartIdx-1},{}}]=trainAndValidationInputData[{{1,validStartIdx-1},{}}]
      trainTargetData[{{1,validStartIdx-1},{}}]=trainAndValidationTargetData[{{1,validStartIdx-1},{}}]
    end
    if(validEndIdx+1 < trainAndValidationInputData:size(1)) then
      trainInputData[{{validStartIdx,trainInputData:size(1)},{}}]=trainAndValidationInputData[{{validEndIdx+1,trainAndValidationInputData:size(1)},{}}]
      trainTargetData[{{validStartIdx,trainTargetData:size(1)},{}}]=trainAndValidationTargetData[{{validEndIdx+1,trainAndValidationTargetData:size(1)},{}}]
    end

    trainAndValidationDataSetup[fold]={trainInput=trainInputData,trainTarget=trainTargetData,validInput=validInputData,validTarget=validTargetData}
  end
  return {trainAndValidationDataSetup=trainAndValidationDataSetup,testInputData=inputTestData,testTargetData=targetTestData,
    trainAndValidationInputData=trainAndValidationInputData,trainAndValidationTargetData=trainAndValidationTargetData}
end

--normalise the columns in the 2D tensor data, columns is array containing columns to normalise 
function Data.normaliseData(data,columns)
  --normalizing input data  
  for _,c in ipairs(columns) do
    data[{{},c}]:add(-data[{{},c}]:mean())
    data[{{},c}]:div(data[{{},c}]:std())
  end
end

return Data
