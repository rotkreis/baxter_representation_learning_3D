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

-- load model
criterion = nn.MSECriterion()
require(MODEL_ARCHITECTURE_FILE) -- minimalnetmodel
Model = getModel(DIMENSION) -- 3
print(Model)
parameters, gradParameters = Model:getParameters()
--criterion

local LR = 0.001
local optimState = {LearningRate = LR}

for epoch = 1, 1000 do
  indice = torch.random(1, NB_SEQUENCES)
  --indice = torch.random(1, NB_SEQUENCES - 1)
  local data = load_seq_by_id(indice)
  assert(data, "Error loading data")
  for batch = 1, 10 do
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
  end
end

for k = 1, 10 do
    --indice = NB_SEQUENCES
    indice = torch.random(1, NB_SEQUENCES)
    data = load_seq_by_id(indice)
    i = torch.random(1, #data.Infos[1])
    input = data.images[i]:double()
    label = torch.Tensor(3)
    label[1] = data.Infos[1][i]
    label[2] = data.Infos[2][i]
    label[3] = data.Infos[3][i]
    print(i)
    print(Model:forward(input))
    print(label)
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

