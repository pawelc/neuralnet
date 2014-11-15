t=require 'torch'

a=t.Tensor({1,2,3,4})
b=a:resize(5)
a=b:narrow(1,1,4)
print(a)
print(b)
