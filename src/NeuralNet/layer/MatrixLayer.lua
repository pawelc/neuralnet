--MatrixLayer is the layer of matrix network implemented with matrices

local t = require 'torch'

local MatrixLayer = {}

--Create the input MatrixLayer of given size, only invoked by concreate layer type
function MatrixLayer:__new (o)
  o = o or {}
  setmetatable(o, self)
  self.__index = self
  return o
end

--Generic functions
function MatrixLayer.checkSize(size)
  if(size<=0) then
    error("size has to be greater than 0")
  end
end

--Set error function
function MatrixLayer:errFun(errFun)
  self.errFun = errFun
  return self
end

function MatrixLayer:setAdjustParamsFun(adjustParamsFun)
  self.adjustParamsFun = adjustParamsFun
  return self
end

return MatrixLayer