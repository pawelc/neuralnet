--Utils for tables

local TableUtil={} 

--converts table to string
function TableUtil.tostring(t)
  buff={}
  i=1
  for k,v in pairs(t) do
    buff[i]=string.format("%s -> %s",k,v)
    i = i + 1
  end
  return table.concat(buff,", ")
end

function TableUtil.shallowCopy(t)
  local t2 = {}
  for k,v in pairs(t) do
    t2[k] = v
  end
  return t2
end

return TableUtil