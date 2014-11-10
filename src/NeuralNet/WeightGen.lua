local t=require("torch")

local function uniformAroundZero(outSize,inSize)
  return (t.rand(outSize,inSize) * 2 - 1) * 1
end
 

return {
  uniformAroundZero=uniformAroundZero
}