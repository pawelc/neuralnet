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

return LuaUtils