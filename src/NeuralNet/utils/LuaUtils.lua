local LuaUtils = {}

function LuaUtils.chainExc(functions)
  return function(...)
    local res=functions[1](...)
    for i = 2,#functions do
      res=functions[i](res)
    end
    return res
  end
end

function LuaUtils.csvStringToTable(s)
  local output={}
  for match in s:gmatch("([^,%s]+)") do
    output[#output + 1] = tonumber(match)
  end
  return output
end

return LuaUtils
