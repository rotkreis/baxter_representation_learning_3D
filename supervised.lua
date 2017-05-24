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
function reinitNet()
  -- reinit weights for cross-validation
  local method = 'xavier'
  Model = require('weight-init')(Model, method)
end

function getLabel(data, index)
  -- get label of image i in data sequence
  local label = torch.Tensor(3)
  label[1] = data.Infos[1][index]
  label[2] = data.Infos[2][index]
  label[3] = data.Infos[3][index]
  return label
end
-- simple evaluation of acurracy
function evaluate(data)
  local n = (#data.images)
  local err = 0
  for i = 1,n do
    local input = data.images[i]:double()
    local truth = getLabel(data, i)
    output = Model:forward(input)
    err = err + (output - truth):pow(2):sum()
  end
  err = err / n
  return err
end



function train(Model, nb_epochs, nb_batches, LR, indice_val)
  -- For simpleData3D at the moment. Training using sequences 1-7, 8 as test.
  -- Given an indice_val, train and return the *errors* on training set as well
  -- as on validation set.
  -- TODO generate logs
  -- TODO perhaps plot graphs (though not for everyone?)
  -- TODO print hyperparameters, or can be done where called
  collectgarbage()
  Model:clearState()
  print('--------------Validation set : '..indice_val..' ---------------')
  xlua.progress(0, nb_epochs)
  logger = optim.Logger('Log/sup'..'ep'..nb_epochs..'ba'..nb_batches..'LR'..LR..'val'..indice_val..'.log')
  logger:setNames{'Validation Accuracy'}
  logger:display(false)

  local optimState = {LearningRate = LR}
  index_train = {}
  for i = 1, NB_SEQUENCES - 1 do
    index_train[i] = i
  end
  table.remove(index_train, indice_val)

  local err_val = torch.Tensor(nb_epochs)
  local err_train = torch.Tensor(nb_epochs)
  for epoch = 1, nb_epochs do
    -- load data sequence
    indice = torch.random(1, #index_train)
    local data = load_seq_by_id(index_train[indice])
    assert(data, "Error loading data")

    for batch = 1, nb_batches do
        n = #data.Infos[1]
        i = torch.random(1, n)
        local input = data.images[i]:double()
        local label = getLabel(data, i)
        -- closure for optim
        local feval = function(x)
            collectgarbage()
            if x ~= parameters then
                parameters:copy(x)
            end
            gradParameters:zero()
            local state = Model:forward(input)
            local loss = criterion:forward(state,label)
            local dloss_dstate = criterion:backward(state, label)
            --print(dloss_dstate)
            Model:backward(input, dloss_dstate)
            return loss, gradParameters
        end
        optim.adagrad(feval, parameters, optimState) -- or adam
    end
    xlua.progress(epoch, nb_epochs)
    err_val[epoch] = evaluate(load_seq_by_id(indice_val))
    logger:add{err_val[epoch]}
    logger:style{'+-'}
    logger:plot()
    print(err_val[epoch])
  -- error on training set, too slow
    -- local avg = 0
    -- for i = 1, #index_train do
    --   avg = avg + evaluate(load_seq_by_id(i))
    -- end
    -- err_train[epoch] = avg / #index_train
  end
  performance_val = err_val[nb_epochs]-- final error
  return performance_val, err_val
end

------- cross-validation ---------------------------
nb_slices = NB_SEQUENCES
nb_epochSet = {30, 50}
nb_batchSet = {10}
lrSet = {0.01, 0.001}
configs = {}
performances = torch.Tensor(#nb_epochSet * #nb_batchSet *  #lrSet)
local count = 1

function cross_validation()
-- K-fold cross-valition on epoch size, batch size, and learning rate
-- computing
  for i, nb_epochs in pairs(nb_epochSet) do
    for j, nb_batches in pairs(nb_batchSet) do
      for k, lr in pairs(lrSet) do
        print("config set:", lr, nb_epochs, nb_batches)
        -- training, K-fold (K = 8)
        -- how to make sure each time a new model?
        -- Model = getModel(DIMENSION)
        reinitNet();
        local avgPerformance = 0
        for indice_val = 1, nb_slices-1 do
           avgPerformance = avgPerformance + train(Model, nb_epochs, nb_batches, lr, indice_val)
        end
        performances[count] = avgPerformance / (nb_slices - 1)
        configs[count] = {'Epoch = '..nb_epochs, 'Batches = '..nb_batches, 'LR = '..lr}
        count = count + 1
      end
    end
  end
-- print errors
  for i = 1, performances:size(1) do
    print("MSE", performances[i], configs[i])
  end
-- pick the best model, apply on test set
end

---------------- single run -----------------
-- training (hyper)parameters
local LR = 0.01
local nb_epochs = 10
local nb_batches = 10
local err = torch.Tensor(nb_epochs)
indice_test = NB_SEQUENCES
local indice_val = 3
_, err = train(Model, nb_epochs, nb_batches, LR, indice_val)
-- gnuplot.pngfigure('supLearn.png')
-- gnuplot.plot({'MSE Loss', err})
-- gnuplot.plotflush()
torch.save('supervised.Model', Model)

cross_validation()

-- plot error on val ----------

---------- intuition ---------------------------
print('-------validation------------')
data = load_seq_by_id(indice_val)
for i = 1,2 do
  local input = data.images[i]:double()
  local truth = getLabel(data, i)
  output = Model:forward(input)
  print('**** ---- ****')
  print('truth')
  print(truth)
  print('output')
  print(output)
end

print('-------training------------')
data = load_seq_by_id(1)
for i = 1,2 do
  local input = data.images[i]:double()
  local truth = getLabel(data, i)
  output = Model:forward(input)
  print('**** ---- ****')
  print('truth')
  print(truth)
  print('output')
  print(output)
end
