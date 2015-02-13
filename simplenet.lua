require('torch')
require('nn')
require('image')
require('xlua')
require('optim')
matio = require('matio')

torch.setnumthreads(1)

local filename = paths.concat('model.net')

-- load data
trainData = matio.load("/vol/temp/sreitsma/training.mat", "data")[{{1,1000},}]
trainLabels = matio.load("/vol/temp/sreitsma/labels.mat", "labels")[1] + 1

-- parameters
nChannels = 1 -- the amount of channels in the input image (1 for Plankton)
nConvolutionMaps = {48, 128, 64} -- the amount of feature maps in layer k
nKernelSize = {9, 5, 3} -- the kernel size in layer k
nKernelSkipSize = {2, 1, 1}
nPoolSize = 3 -- the pool kernel size
nPoolSkipSize = 2 -- the pool kernel skip size
nOutputs = trainLabels[trainLabels:size(1)]
nDataSet = trainData:size(1)
nBatchSize = 100

-- Container:
model = nn.Sequential()
criterion = nn.ClassNLLCriterion()

-- Set up AlexNet
local features = nn.Concat(1)

local branch1 = nn.Sequential() -- branch 1
branch1:add(nn.SpatialConvolutionMM(nChannels, nConvolutionMaps[1], nKernelSize[1], nKernelSize[1], nKernelSkipSize[1], nKernelSkipSize[1], 2, 2))
branch1:add(nn.ReLU(true))
branch1:add(nn.SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))
branch1:add(nn.SpatialConvolutionMM(nConvolutionMaps[1], nConvolutionMaps[2], nKernelSize[2], nKernelSize[2], nKernelSkipSize[2], nKernelSkipSize[2], 2, 2))
branch1:add(nn.ReLU(true))
branch1:add(nn.SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))
branch1:add(nn.SpatialConvolutionMM(nConvolutionMaps[2], nConvolutionMaps[3], nKernelSize[3], nKernelSize[3], nKernelSkipSize[3], nKernelSkipSize[3], 1, 1))
branch1:add(nn.ReLU(true))
--branch1:add(nn.SpatialConvolutionMM(nConvolutionMaps[3], nConvolutionMaps[4], nKernelSize[4], nKernelSize[4], nKernelSkipSize[4], nKernelSkipSize[4], 1, 1))
--branch1:add(nn.ReLU(true))
--branch1:add(nn.SpatialConvolutionMM(nConvolutionMaps[4], nConvolutionMaps[5], nKernelSize[5], nKernelSize[5], nKernelSkipSize[5], nKernelSkipSize[5], 1, 1))
--branch1:add(nn.ReLU(true))
branch1:add(nn.SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))

local branch2 = branch1:clone() -- branch 2
for k,v in ipairs(branch2:findModules('nn.SpatialConvolutionMM')) do
  v:reset() -- reset branch 2's weights
end

features:add(branch1)
--features:add(branch2)

local classifier = nn.Sequential()
classifier:add(nn.View(64*2*2))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(64*2*2, 4096))
classifier:add(nn.Threshold(0, 1e-6))
--classifier:add(nn.Dropout(0.5))
--classifier:add(nn.Linear(4096, 4096))
--classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, nOutputs))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential():add(features):add(classifier)

if arg[1] == "resume" then
	model = torch.load(filename)
end

parameters,gradParameters = model:getParameters()

optimState = {
  learningRate = 1e-3,
  weightDecay = 0,
  momentum = 0,
  learningRateDecay = 1e-7
}
optimMethod = optim.sgd

function train()
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(nDataSet)

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. nBatchSize .. ']')
   for t = 1,trainData:size(1),nBatchSize do
      -- disp progress
      xlua.progress(t, trainData:size(1))

      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+nBatchSize-1,trainData:size(1)) do
         -- load new sample
         local input = trainData[shuffle[i]]
         local target = trainLabels[shuffle[i]]
         if not cuda then input = input:double()
         else input = input:cuda() end
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
			  --confusion:add(output, targets[i])
		   end

		   -- normalize gradients and f(X)
		   gradParameters:div(#inputs)
		   f = f/#inputs

		   -- return f and df/dX
		   return f,gradParameters
	  end

      -- optimize on current mini-batch
      optimMethod(feval, parameters, optimState)
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size(1)
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- save/log current net
   print('==> saving model to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1
end

while true do
	train()
end