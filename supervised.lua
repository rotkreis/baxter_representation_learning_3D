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
-- THIS IS WHERE ALL THE CONSTANTS SHOULD COME FROM
-- See const.lua file for more details
require 'const'
-- try to avoid global variable as much as possible

local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES = #list_folders_images

-- model
criterion = nn.MSECriterion()
      -- Model = torch.load(MODEL_FILE_STRING):double()
require(MODEL_ARCHITECTURE_FILE) -- minimalnetmodel
Model = getModel(DIMENSION) -- 3
print(Model)
parameters, gradParameters = Model:getParameters()


-- simple evaluation, not using correlation for the moment, a sequence
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
local optimState = {LearningRate = LR}
local nb_epochs = 300
local nb_batches = 10
local err = torch.Tensor(nb_epochs)

print('--------------Epoch : '..nb_epochs..' ---------------')
xlua.progress(0, nb_epochs)
for epoch = 1, nb_epochs do
  indice = torch.random(2, NB_SEQUENCES)
  --indice = torch.random(1, NB_SEQUENCES - 1)
  local data = load_seq_by_id(indice)
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
  err[epoch] = evaluate(load_seq_by_id(1))
  print(err[epoch])
end
-- print(err)

data = load_seq_by_id(1)
local err = 0
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
  -- err = err + (output - truth):pow(2):sum() / truth:pow(2):sum()
end
-- --load data
--local indice1 = 8
--local data1 = load_seq_by_id(indice1)
--print(#data1.images)
--label = torch.Tensor(#data1.Infos[1],3)
--label[{{}, 1}] = torch.DoubleTensor(data1.Infos[1])
--label[{{}, 2}] = torch.DoubleTensor(data1.Infos[2])
--label[{{}, 3}] = torch.DoubleTensor(data1.Infos[3])
--print(data1.Infos[1])
