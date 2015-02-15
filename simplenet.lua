require('torch')
require('nn')
require('image')
require('xlua')
require('optim')
require('cudnn')
require('cunn')
matio = require('matio')

-- parse command line arguments
if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options')
	cmd:option('-cuda', false, 'whether to use cuda')
	cmd:option('-threads', 1, 'amount of CPU threads to use, not relevant for cuda')
	cmd:option('-resume', false, 'whether to resume loading from model.net file')
	cmd:text()
	opt = cmd:parse(arg or {})
end

-- set global Torch parameters
torch.setnumthreads(opt.threads)

-- check for CUDA
if opt.cuda then
	SpatialConvolution = cudnn.SpatialConvolution
	ReLU = cudnn.ReLU
	SpatialMaxPooling = cudnn.SpatialMaxPooling
else
	SpatialConvolution = nn.SpatialConvolutionMM
	ReLU = nn.ReLU
	SpatialMaxPooling = nn.SpatialMaxPooling
end

-- load data
local trainData = matio.load("data/training.mat", "data") -- to use a subset, append [{{1,1000},}]
local trainLabels = matio.load("data/labels.mat", "labels")[1] + 1 -- add one because lua assumes 1 as starting index
local filename = 'model.net'

print(trainData[1])

-- parameters
local nChannels = 1 -- the amount of channels in the input image (1 for Plankton)
local nConvolutionMaps = {64, 128, 64} -- the amount of feature maps in layer k
local nKernelSize = {9, 5, 3} -- the kernel size in layer k
local nKernelSkipSize = {1, 1, 2} -- the stride
local nPoolSize = 3 -- the pool kernel size
local nPoolSkipSize = 2 -- the pool kernel stride
local nOutputs = trainLabels[trainLabels:size(1)]
local nBatchSize = 256 -- power of 2 for faster processing

-- Container:
local model = nn.Sequential()
local criterion = nn.ClassNLLCriterion()

-- Set up network
local branch1 = nn.Sequential()
branch1:add(SpatialConvolution(nChannels, nConvolutionMaps[1], nKernelSize[1], nKernelSize[1], nKernelSkipSize[1], nKernelSkipSize[1]))
branch1:add(ReLU(true))
branch1:add(SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))
branch1:add(SpatialConvolution(nConvolutionMaps[1], nConvolutionMaps[2], nKernelSize[2], nKernelSize[2], nKernelSkipSize[2], nKernelSkipSize[2]))
branch1:add(ReLU(true))
branch1:add(SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))
branch1:add(SpatialConvolution(nConvolutionMaps[2], nConvolutionMaps[3], nKernelSize[3], nKernelSize[3], nKernelSkipSize[3], nKernelSkipSize[3]))
branch1:add(ReLU(true))
branch1:add(SpatialMaxPooling(nPoolSize, nPoolSize, nPoolSkipSize, nPoolSkipSize))

local classifier = nn.Sequential()
classifier:add(nn.View(64*2*2))
classifier:add(nn.Linear(64*2*2, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, nOutputs))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential():add(branch1):add(classifier)

-- check for resume
if opt.resume then
	model = torch.load(filename)
end

-- check for cuda
if opt.cuda then
	model:cuda()
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
	shuffle = torch.randperm(trainData:size(1))

	-- do one epoch
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [batchSize = ' .. nBatchSize .. ']')
	for t = 1,trainData:size(1),nBatchSize do
		-- disp progress
		xlua.progress(t-1, trainData:size(1))

		-- create mini batch
		local inputs

		if opt.cuda then
			inputs = torch.CudaTensor(nBatchSize, nChannels, trainData:size(3), trainData:size(4))
			targets = torch.CudaTensor(nBatchSize)
		else
			inputs = torch.Tensor(nBatchSize, nChannels, trainData:size(3), trainData:size(4))
			targets = torch.Tensor(nBatchSize)
		end

		local tensor_index = 1
		for i = t,math.min(t+nBatchSize-1,trainData:size(1)) do
			-- load new sample
			local input = trainData[shuffle[i]]
			local target = trainLabels[shuffle[i]]
			if not opt.cuda then input = input:double()
			else input = input:cuda() end
			inputs[tensor_index] = input
			targets[tensor_index] = target
			tensor_index = tensor_index + 1
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

			local outputs = model:forward(inputs)
			outputs = outputs:double()

			local err = criterion:forward(outputs, targets)
			local df_do = criterion:backward(outputs, targets)

			f = f + err

			if opt.cuda then
				df_do = df_do:cuda()
			end

			model:backward(inputs, df_do)

			-- normalize gradients and f(X)
			gradParameters:div(targets:size(1))
			print(model:forward(trainData[1]:cuda()))

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
	if epoch % 10 == 0 then
		print('==> saving model to '..filename)
		torch.save(filename, model)
	end

	-- next epoch
	epoch = epoch + 1
end

while true do
	train()
end
