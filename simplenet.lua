require('torch')
require('nn')
require('image')
require('xlua')
require('optim')
require('cudnn')
require('cunn')
Plot = require('itorch.Plot')
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
	cmd:option('-subset', false, 'whether to run on a subset of the data for testing')
	cmd:option('-save', false, 'whether to save the model each 10 epochs')
	cmd:option('-test', false, 'whether to split the training set 90/10 and test during training')
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
local trainDataFull = matio.load("data/training.mat", "data")
local trainLabelsFull = matio.load("data/labels.mat", "labels")[1] + 1 -- add one because lua assumes 1 as starting index

if opt.subset then
	trainDataFull = trainDataFull[{{1,2048},}]
	trainLabelsFull = trainLabelsFull[{{1,2048},}]
end

local trainData
local trainLabels
local testData
local testLabels

if opt.test then
	shuffle = torch.randperm(trainDataFull:size(1))
	nTraining = math.floor(trainDataFull:size(1) * 0.90)
	nTesting = trainDataFull:size(1) - nTraining

	trainData = torch.Tensor(nTraining, trainDataFull:size(2), trainDataFull:size(3), trainDataFull:size(4))
	trainLabels = torch.Tensor(nTraining)
	testData = torch.Tensor(nTesting, trainDataFull:size(2), trainDataFull:size(3), trainDataFull:size(4))
	testLabels = torch.Tensor(nTesting)

	for t = 1,trainDataFull:size(1) do
		sample = trainDataFull[shuffle[t]]
		label = trainLabelsFull[shuffle[t]]

		if t <= nTraining then
			trainData[t] = sample
			trainLabels[t] = label
		else
			testData[t-nTraining] = sample
			testLabels[t-nTraining] = label
		end
	end
else
	trainData = trainDataFull
	trainLabels = trainLabelsFull
end

local filename = 'model.net'

-- parameters
local nChannels = 1 -- the amount of channels in the input image (1 for Plankton)
local nOutputs = trainLabelsFull[trainLabelsFull:size(1)]
local nBatchSize = 32 -- power of 2 for faster processing

-- free up memory
trainDataFull = nil
trainLabelsFull = nil

-- Container:
local model = nn.Sequential()
local criterion = nn.ClassNLLCriterion()

-- Set up OverFeat network
local features = nn.Sequential()

features:add(SpatialConvolution(1, 96, 7, 7, 4, 4)) -- (64 - 7 + 1)/4 = 15
features:add(ReLU(true))
features:add(SpatialMaxPooling(2, 2, 1, 1)) -- 14

features:add(SpatialConvolution(96, 256, 5, 5, 1, 1)) -- (14 - 5 + 1)/1 = 10
features:add(ReLU(true))
features:add(SpatialMaxPooling(2, 2, 1, 1)) -- 9

features:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- (9 - 3 + 3)/1 = 9
features:add(ReLU(true))

features:add(SpatialMaxPooling(2, 2, 2, 2)) -- 4

local classifier = nn.Sequential()
classifier:add(nn.View(512*4*4))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(512*4*4, 3072))
classifier:add(nn.Threshold(0, 1e-6))

classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(3072, 4096))
classifier:add(nn.Threshold(0, 1e-6))

classifier:add(nn.Linear(4096, nOutputs))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential():add(features):add(classifier)

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
	learningRate = 5e-2,
	weightDecay = 1e-5,
	momentum = 0.6,
	learningRateDecay = 5e-7
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

	local totalError = 0
	local nBatches = 0

	for t = 1,trainData:size(1),nBatchSize do
		-- disp progress
		if (t-1) % 3200 == 0 then
			xlua.progress(t-1, trainData:size(1))
		end

		-- create mini batch
		local inputs
		local targets
		local actualBatchSize = math.min(t+nBatchSize-1,trainData:size(1)) - t + 1

		if opt.cuda then
			inputs = torch.CudaTensor(actualBatchSize, nChannels, trainData:size(3), trainData:size(4))
			targets = torch.CudaTensor(actualBatchSize)
		else
			inputs = torch.Tensor(actualBatchSize, nChannels, trainData:size(3), trainData:size(4))
			targets = torch.Tensor(actualBatchSize)
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

			local outputs = model:forward(inputs)
			outputs = outputs:double()

			local err = criterion:forward(outputs, targets)
			local df_do = criterion:backward(outputs, targets)

			if opt.cuda then
				df_do = df_do:cuda()
			end

			model:backward(inputs, df_do)

			totalError = totalError + err

			-- normalize gradients and f(X)
			gradParameters:div(targets:size(1))

			return err,gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)

		nBatches = nBatches + 1
	end

	-- time taken
	time = sys.clock() - time
	time = time / trainData:size(1)
	print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms, total error = ' .. (totalError / nBatches))

	-- save/log current net
	if epoch % 10 == 0 and opt.save then
		print('==> saving model to '..filename)
		torch.save(filename, model)
	end

	-- next epoch
	epoch = epoch + 1

	return totalError / nBatches
end

function test(subset)
	epoch = epoch or 1

	local totalError = 0
	local amount

	if subset then
		amount = math.min(512, nTesting)
	else
		amount = nTesting
	end

	for t = 1,amount do
		if t % 512 == 0 then
			xlua.progress(t-1, amount)
		end

		local input = testData[t]
		input = input:cuda()
		local output = model:forward(input)
		output = output:double()

		local err = criterion:forward(output, testLabels[t])
		totalError = totalError + err
	end

	return totalError / amount
end

local train_error_rates = {}
local test_error_rates = {}
while true do
	table.insert(train_error_rates, train())

	if opt.test then
		table.insert(test_error_rates, test())
	end

	-- plot that shit
	plot = Plot():line(torch.range(1,#train_error_rates), train_error_rates,'red','Training set error'):legend(true):title('Error rate')
	if opt.test then
		plot:line(torch.range(1,#test_error_rates), test_error_rates,'blue','Test set error'):draw()
	end
	plot:save('out.html')
end