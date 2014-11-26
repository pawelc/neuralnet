TorchUtils={}

--copied from https://github.com/rlowrance/re/blob/master/argmax.lua
local function argmax_1D(v)
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

local function argmax_2D(matrix)
   local nRows = matrix:size(1)
   local result = torch.Tensor(nRows)
   for i = 1, nRows do
      result[i] = argmax_1D(matrix[i])
   end
   return result
end

-- index of largest element
-- ARGS:
-- tensor : 1D or 2D Tensor
-- RETURNS:
-- result : scalar (if v is 1D) or 1D Tensor (if v is 2D)
--          if scalar i : integer in [1, v:size(1)] such that v[i] >= v[k] for all k
--          if 1D Tensor, then the scalar i for each row
function TorchUtils.argmax(tensor)
   local nDimension = tensor:nDimension()
   if nDimension == 1 then
      return argmax_1D(tensor)
   elseif nDimension == 2 then
      return argmax_2D(tensor)
   else
      error('tensor has %d dimensions, not 1 or 2', nDimension)
   end
end

--Returns tensor as string of comma delimited elements in the row and rows separated by new line
function TorchUtils.tensorToCsv(tensor)
   local nDimension = tensor:nDimension()
   local lines={}
   if nDimension == 2 then
      for i = 1,tensor:size(1) do
        lines[#lines+1]=TorchUtils.vecToCsv(tensor[{i,{}}])  
      end
   else
      error(string.format('tensor has %d dimensions, not 2', nDimension))
   end
   return table.concat(lines,"\n")
end

--Returns 2D lattice as CSV where first two columns are localtions of neurons
function TorchUtils.twoDimLatticeToCsv(tensor)
   local nDimension = tensor:nDimension()
   local lines={}
   if nDimension == 3 then
      for x = 1,tensor:size(1) do
        for y = 1,tensor:size(2) do 
          lines[#lines+1]=x..","..y..","..TorchUtils.vecToCsv(tensor[x][y])
        end  
      end
   else
      error(string.format('tensor has %d dimensions, not 2', nDimension))
   end
   return table.concat(lines,"\n")
end

--Returns vector as string of comma delimited elements
function TorchUtils.vecToCsv(vec)
   local nDimension = vec:nDimension()
   local elements={}
   if nDimension == 1 then
      for i = 1,vec:size(1) do
        elements[#elements+1]=vec[i] 
      end
   else
      error('vector has %d dimensions, not 1', nDimension)
   end
   return table.concat(elements,",")
end

return TorchUtils