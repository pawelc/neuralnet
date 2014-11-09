require("nn")
require("optim")

trsize = 4

trainData={
  size=function()
    return 4
  end,  
  data=torch.Tensor({
    {-1,1},
    {1,-1},
    {-1,-1},
    {1,1}
  }),
  labels=torch.Tensor({
  {2},
  {2},
  {1},
  {1}
  })
}

model = nn.Sequential()
--model:add(nn.Reshape(ninputs))
model:add(nn.Linear(2,2))
model:add(nn.Tanh())
model:add(nn.Linear(2,1))
model:add(nn.Tanh())

criterion = nn.MSECriterion()
criterion.sizeAverage = false

-- classes
classes = {'-1','1'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

output = "output"
-- Log results to files
trainLogger = optim.Logger(paths.concat(output, 'train.log'))
testLogger = optim.Logger(paths.concat(output, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
parameters,gradParameters = model:getParameters()

optimState = {
    learningRate = 1e-3,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
 }
 optimMethod = optim.sgd
 
 batchSize = 1 --mini-batch size (1 = pure stochastic)
optType = "double"

function train()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trsize)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
   for t = 1,trainData:size(),batchSize do
      -- disp progress
      xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if optType == 'double' then input = input:double()
         elseif optType == 'cuda' then input = input:cuda() end
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
                       -- get new parameters
                       if x ~= parameters then
                          parameters:copy(x)
                       end

                       -- reset gradients
                       gradParameters:zero()

                       -- f is the average of all criterions
                       local f = 0

                       -- evaluate function for complete mini batch
                       for i = 1,#inputs do
                          -- estimate f
                          local output = model:forward(inputs[i])
                          local err = criterion:forward(output, targets[i])
                          f = f + err

                          -- estimate df/dW
                          local df_do = criterion:backward(output, targets[i])
                          model:backward(inputs[i], df_do)

                          -- update confusion
                          confusion:add(output, targets[i])
                       end

                       -- normalize gradients and f(X)
                       gradParameters:div(#inputs)
                       f = f/#inputs

                       -- return f and df/dX
                       return f,gradParameters
                    end

      -- optimize on current mini-batch
      if optimMethod == optim.asgd then
         _,_,average = optimMethod(feval, parameters, optimState)
      else
         optimMethod(feval, parameters, optimState)
      end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

   -- save/log current net
   local filename = paths.concat(output, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end

--while true do
  train()
  print("checking")
  print(model:forward(torch.Tensor({-1,1})))
  print(model:forward(torch.Tensor({1,-1})))
  print(model:forward(torch.Tensor({-1,-1})))
  print(model:forward(torch.Tensor({1,1})))
--end