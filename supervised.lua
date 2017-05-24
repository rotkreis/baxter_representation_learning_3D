require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
--require 'cunn'
require 'nngraph'
require 'MSDC'
require 'functions'
require 'printing'
require "Get_Images_Set"
require 'optim_priors'
require 'definition_priors'
require 'gnuplot'
require 'os'
require 'paths'

-- THIS IS WHERE ALL THE CONSTANTS SHOULD COME FROM
-- See const.lua file for more details
require 'const'
-- try to avoid global variable as much as possible

local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES = #list_folders_images

---------------- model & criterion ---------------------------
-- TODO load modle using minimalNetModel (now )
require(MODEL_ARCHITECTURE_FILE) -- minimalnetmodel
Model = getModel(DIMENSION) -- DIMENSION = 3
parameters, gradParameters = Model:getParameters()
criterion = nn.MSECriterion()


-- simple evaluation
function evaluate(data)
  local n = (#data.images)
  local err = 0
  for i = 1,n do
    local input = data.images[i]:double()
    local truth = torch.Tensor(3)
    truth[1] = data.Infos[1][i]
    truth[2] = data.Infos[2][i]
    truth[3] = data.Infos[3][i]
    output = Model:forward(input)
    -- err = err + (output - truth):pow(2):sum() / truth:pow(2):sum()
    err = err + (output - truth):pow(2):sum()
  end
  err = err / n
  return err
end


-- training
local LR = 0.001
local nb_epochs = 50
local nb_batches = 20
local err = torch.Tensor(nb_epochs)

-- train & test
indice_test = NB_SEQUENCES

function train(nb_epochs, nb_batches, LR, indice_val)
  collectgarbage()
  Model:clearState()
  -- print('--------------Epoch : '..nb_epochs..' ---------------')
  print('--------------indice_val : '..indice_val..' ---------------')
  xlua.progress(0, nb_epochs)
  local optimState = {LearningRate = LR}

  index_train = {}
  for i = 1, NB_SEQUENCES - 1 do
    index_train[i] = i
  end
  table.remove(index_train, indice_val)

  local err = torch.Tensor(nb_epochs)
  for epoch = 1, nb_epochs do
    --indice = torch.random(1, NB_SEQUENCES - 1)
    indice = torch.random(1, #index_train)
    local data = load_seq_by_id(index_train[indice])
    assert(data, "Error loading data")
    for batch = 1, nb_batches do
        n = #data.Infos[1]
        i = torch.random(1, n)
        input = data.images[i]:double()
        label = torch.Tensor(3)
        label[1] = data.Infos[1][i]
        label[2] = data.Infos[2][i]
        label[3] = data.Infos[3][i]

        local feval = function(x)
            collectgarbage()
            if x ~= parameters then
                parameters:copy(x)
            end
            -- calculation
            gradParameters:zero()
            local state = Model:forward(input)
            local loss = criterion:forward(state,label)
            local dloss_dstate = criterion:backward(state, label)
            --print(dloss_dstate)
            Model:backward(input, dloss_dstate)
            return loss, gradParameters
        end
        optim.adagrad(feval, parameters, optimState)
        -- optim.adam(feval, parameters, optimState)
    end
    xlua.progress(epoch, nb_epochs)
    err[epoch] = evaluate(load_seq_by_id(indice_val))
    print(err[epoch])
  end
  performance = evaluate(load_seq_by_id(indice_val)) -- final error
  return performance, err  -- on validation set only, TODO add training set
end

-- cross-validation ---------------------------
-- lrSet = {0.1, 0.01, 0.001, 0.0001}
nb_slices = NB_SEQUENCES
nb_epochSet = {10, 20, 100, 200, 500}
--nb_epochSet = {10, 20}
nb_batchSet = {10, 20}
lrSet = {0.01}
configs = {}
performances = torch.Tensor(#nb_epochSet * #nb_batchSet *  #lrSet)
local count = 1
--for i, nb_epochs in pairs(nb_epochSet) do
  --for j, nb_batches in pairs(nb_batchSet) do
    --for k, lr in pairs(lrSet) do
      --print("config set:", lr, nb_epochs, nb_batches)
      ---- training, K-fold (K = 8)
      --local avgPerformance = 0
      --for indice_val = 1, nb_slices-1 do
         --avgPerformance = avgPerformance + train(nb_epochs, nb_batches, lr, indice_val)
      --end
      --performances[count] = avgPerformance / (nb_slices - 1)
      --configs[count] = {'Epoch = '..nb_epochs, 'Batches = '..nb_batches, 'LR = '..lr}
      --count = count + 1
    --end
  --end
--end


-- pick the best model, apply on test set

--for i = 1, performances:size(1) do
  --print("MSE", performances[i], configs[i])
--end

-- plot error on val
local indice_val = 2
_, err = train(nb_epochs, nb_batches, LR, indice_val)
gnuplot.pngfigure('supLearn.png')
gnuplot.plot({'MSE Loss', err})
gnuplot.plotflush()
torch.save('supervised.Model', Model)
print(err)

-- intuition ---------------------------
print('validation')
data = load_seq_by_id(indice_val)
for i = 1,5 do
  local input = data.images[i]:double()
  local truth = torch.Tensor(3)
  truth[1] = data.Infos[1][i]
  truth[2] = data.Infos[2][i]
  truth[3] = data.Infos[3][i]
  output = Model:forward(input)
  print('pair')
  print('truth')
  print(truth)
  print('output')
  print(output)
end

print('training')
data = load_seq_by_id(indice_val)
for i = 1,5 do
  local input = data.images[i]:double()
  local truth = torch.Tensor(3)
  truth[1] = data.Infos[1][i]
  truth[2] = data.Infos[2][i]
  truth[3] = data.Infos[3][i]
  output = Model:forward(input)
  print('pair')
  print('truth')
  print(truth)
  print('output')
  print(output)
end
  -- err = err + (output - truth):pow(2):sum() / truth:pow(2):sum()

-- --load data ----------------------------
--local indice1 = 8
--local data1 = load_seq_by_id(indice1)
--print(#data1.images)
--label = torch.Tensor(#data1.Infos[1],3)
--label[{{}, 1}] = torch.DoubleTensor(data1.Infos[1])
--label[{{}, 2}] = torch.DoubleTensor(data1.Infos[2])
--label[{{}, 3}] = torch.DoubleTensor(data1.Infos[3])
--print(data1.Infos[1])
