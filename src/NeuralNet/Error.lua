local Error={}

require "torch"

function Error.simple(expected,signal)
  return expected - signal
end

function Error.classification(expected,signal)
  return math.min(1,torch.sum(torch.ne(signal,expected)))
end
 

return Error