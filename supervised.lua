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

function printSamples(indice, n)
  data = load_seq_by_id(indice)
  for i = 1,n do
    local input = data.images[i]:double()
    local truth = getLabel(data, i)
    output = Model:forward(input)
    print('**** ---- ****')
    print('truth')
    print(truth)
    print('output')
    print(output)
  end
end
---------------- model & criterion ---------------------------
-- TODO load modle using minimalNetModel (now )
require(MODEL_ARCHITECTURE_FILE) -- minimalnetmodel
Model = getModel(DIMENSION_OUT) -- DIMENSION = 3
parameters, gradParameters = Model:getParameters()
criterion = nn.MSECriterion()
function reinitNet()
  -- reinit weights for cross-validation
  local method = 'xavier'
  Model:clearState()
  Model:reset()
  Model = require('weight-init')(Model, method)
end

function getLabel(data, index)
  -- get label of image i in data sequence
  local label = torch.Tensor(DIMENSION_IN)
  for i = 1, DIMENSION_IN do
    label[i] = data.Infos[i][index]
  end
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



function train(nb_epochs, nb_batches, LR, indice_val, verbose, final)
  -- For simpleData3D at the moment. Training using sequences 1-7, 8 as test.
  -- Given an indice_val, train and return the *errors* on training set as well
  -- as on validation set.
  -- Can be made general directly to mobileRobot?
  -- TODO generate logs
  -- TODO perhaps plot graphs (though not for everyone?)
  -- TODO print hyperparameters, or can be done where called
  collectgarbage()
  local final = final or 0
  Model:clearState()
  -- print('--------------Validation set : '..indice_val..' ---------------')
  if verbose == 1 then xlua.progress(0, nb_epochs) end
  logger = optim.Logger('Log/' ..DATA_FOLDER..'sup_ep'..nb_epochs..'ba'..nb_batches..'LR'..LR..'val'..indice_val..'.log')
  logger:setNames{'Validation Accuracy'}
  if verbose == 0 then logger:display(false) end

  local optimState = {LearningRate = LR}
  -- i = NB_SEQUENCES is the test set
  index_train = {}
  for i = 1, NB_SEQUENCES - 1 do
    index_train[i] = i
  end
  -- if final evaluation
  if final == 1 then index_train[NB_SEQUENCES] = NB_SEQUENCES end
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
    err_val[epoch] = evaluate(load_seq_by_id(indice_val))
    logger:add{err_val[epoch]}
    logger:style{'+-'}
    logger:plot()
    if verbose == 1 then print(err_val[epoch]) end
    if verbose == 1 then xlua.progress(epoch, nb_epochs) end
  end
  performance_val = err_val[nb_epochs]-- final error
  return performance_val, err_val
end

------- cross-validation ---------------------------

function cross_validation()
-- K-fold cross-valition on epoch size, batch size, and learning rate
  K = NB_SEQUENCES - 1
  nb_epochSet = {35}
  nb_batchSet = {10, 20, 30}
  lrSet = {0.01, 0.001, 0.0001}
  configs = {}
  nb_config = #nb_epochSet * #nb_batchSet *  #lrSet
  performances = torch.Tensor(nb_config)
  local count = 1
  print("iterating over configs")
  xlua.progress(0, nb_config)
-- computing
  for i, nb_epochs in pairs(nb_epochSet) do
    for j, nb_batches in pairs(nb_batchSet) do
      for k, lr in pairs(lrSet) do
        -- training, K-fold
        -- how to make sure each time a new model? DONE, cf reinitNet()
        reinitNet();
        local avgPerformance = 0
        for indice_val = 1, K do
           avgPerformance = avgPerformance + train(nb_epochs, nb_batches, lr, indice_val, 0, 0)
        end
        performances[count] = avgPerformance / K
        configs[count] = {'Epoch = '..nb_epochs, 'Batches = '..nb_batches, 'LR = '..lr}
        print(performances[count])
        print(configs[count])
        count = count + 1
        xlua.progress(count - 1, nb_config)
      end
    end
  end
-- print errors
  -- for i = 2, performances:size(2) do
  --   print("MSE", performances[i], configs[i])
  -- end
-- pick the best model, apply on test set
  min, index = torch.min(performances, 1)
  print("best-model", configs[index[1]])
  torch.save('results/sup.t7', configs[index[1]])
end

---------------- single run -----------------
-- (hyper)parameters for training
local LR = 0.001
local nb_epochs = 50
local nb_batches = 30
local err = torch.Tensor(nb_epochs)
-- local indice_val = NB_SEQUENCES
local indice_val = NB_SEQUENCES
print(parameters:sum())
_, err = train(nb_epochs, nb_batches, LR, indice_val, 1, 1)
print(evaluate(load_seq_by_id(indice_val)))
print("results from"..indice_val.."is")
printSamples(indice_val, 3)
print(parameters:sum())
reinitNet()
print("after reinitiation")
print(parameters:sum())

-- torch.save('supervised.Model', Model)

------------- cross_validation ------------------
-- cross_validation()


---------- intuition ---------------------------
-- print('-------validation------------')
-- printSamples(1, 3)
-- print('-------training------------')
-- printSamples(1, 3)
-- reinitNet()
-- _, err = train(Model, nb_epochs, nb_batches, LR, indice_val)
