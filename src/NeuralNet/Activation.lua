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
 

return {
  sigmAct={fun=sigmAct,funD=sigmActDeriv},
  tanhAct={fun=tanhAct,funD=tanhActDeriv}
}