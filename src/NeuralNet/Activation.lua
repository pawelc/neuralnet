local t = require("torch")

--Sigmoid activation
local function sigmAct (signal)
  return (t.exp(-signal) + 1):pow(-1)
end

--Sigmoid derivative
local function sigmActDeriv (signal)
  return t.cmul(signal,-signal+1)
end

--Tanh activation
local function tanhAct (signal)
  return t.tanh(signal)
end

--Tanh derivative
local function tanhActDeriv (signal)
  return t.cmul(-signal+1,signal+1)
end

--Linear activation
local function linAct (signal)
  return signal
end

--Tanh derivative
local function linActDeriv (signal)
  return t.Tensor(signal:size()):fill(1)
end
 

return {
  sigmAct={fun=sigmAct,funD=sigmActDeriv},
  tanhAct={fun=tanhAct,funD=tanhActDeriv},
  linAct={fun=linAct,funD=linActDeriv}
}