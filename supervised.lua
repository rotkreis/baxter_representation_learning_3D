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

function load_Data()
    -- how to load data
    local indice1 = 8
    local indice2 = 2
    local data1 = load_seq_by_id(indice1)
    local data2 = load_seq_by_id(indice2)
    print(data1)
    print(data1.Infos[1][2])
    print(data1.Infos.reward)
end

function train_Epoch(Models, LOG_FOLDER, LR)
    local nb_batch = 10
    print(NB_SEQUENCES..': sequences')

local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES = #list_folders_images
load_Data()
LR = 0.005
model_file='../models/minimalNetModel'
Model = getModel()
