local t=require("torch")
local WeightGen={}

function WeightGen.uniformAroundZero(outSize,inSize)
  return (t.rand(outSize,inSize) * 2 - 1) * 1
end

function WeightGen.normalise(tensor2d)
  if tensor2d:dim() ~= 2 then
    error("supports only 2D tensors")
  end
  for row = 1,tensor2d:size(1) do
    tensor2d[{{row},{}}] = torch.div(tensor2d[{{row},{}}] ,math.sqrt(torch.dot(tensor2d[{row,{}}] ,tensor2d[{row,{}}] )))  
  end
  return tensor2d
end
 

return WeightGen