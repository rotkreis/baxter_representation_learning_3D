require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
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

function Rico_Training(Models,Mode,data1,data2,criterion,coef,LR,BATCH_SIZE, USE_CONTINUOUS)
   local mom=0.9
   local coefL2=0,0

   if USE_CONTINUOUS then
     batch, action1, action2 = getRandomBatchFromSeparateListContinuous(data1,data2,BATCH_SIZE,Mode)
   else
     batch = getRandomBatchFromSeparateList(data1,data2,BATCH_SIZE,Mode)
   end
   -- create closure to evaluate f(X) and df/dX
   local feval = function(x)
      -- just in case:
      collectgarbage()

      -- get new parameters
      if x ~= parameters then
         parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()
      if Mode=='Simpl' then print("Simpl")
      elseif Mode=='Temp' then loss,grad=doStuff_temp(Models,criterion, batch,coef)
      elseif Mode=='Prop' then
        if USE_CONTINUOUS then
           loss,grad=doStuff_Prop_continuous(Models,criterion,batch,coef, action1, action2)
        else
           loss,grad=doStuff_Prop(Models,criterion,batch,coef)
        end
      elseif Mode=='Caus' then
         if USE_CONTINUOUS then
            loss,grad=doStuff_Caus_continuous(Models,criterion,batch,coef, action1, action2)
         else
            loss,grad=doStuff_Caus(Models,criterion,batch,coef)
         end
      elseif Mode=='Rep' then
        if USE_CONTINUOUS then
           loss,grad=doStuff_Rep_continuous(Models,criterion,batch,coef, action1, action2)
        else
           loss,grad=doStuff_Rep(Models,criterion,batch,coef)
        end
      else print("Wrong Mode")
      end
      return loss,gradParameters
   end
   --sgdState = sgdState or { learningRate = LR, momentum = mom,learningRateDecay = 5e-7,weightDecay=coefL2 }
   --parameters, loss=optim.sgd(feval, parameters, sgdState)
   optimState={learningRate=LR}

   if SGD_METHOD == 'adagrad' then
      parameters,loss=optim.adagrad(feval,parameters,optimState)
   else
      parameters,loss=optim.adam(feval,parameters,optimState)
   end

   -- loss[1] table of one value transformed in just a value
   -- grad[1] we use just the first gradient to print the figure (there are 2 or 4 gradient normally)
   return loss[1], grad
end

function train_Epoch(Models,Prior_Used,LOG_FOLDER,LR, USE_CONTINUOUS)
    local NB_BATCHES= math.ceil(NB_SEQUENCES*90/BATCH_SIZE/(4+4+2+2))
    --90 is the average number of images per sequences, div by 12 because the network sees 12 images per iteration
    -- (4*2 for rep and prop, 2*2 for temp and caus)

    local REP_criterion=get_Rep_criterion()
    local PROP_criterion=get_Prop_criterion()
    local CAUS_criterion=get_Caus_criterion()
    local TEMP_criterion=nn.MSDCriterion()

    local Temp_loss_list, Prop_loss_list, Rep_loss_list, Caus_loss_list = {},{},{},{}
    local Temp_loss_list_test,Prop_loss_list_test,Rep_loss_list_test,Caus_loss_list_test = {},{},{},{}
    local Sum_loss_train, Sum_loss_test = {},{}
    local Temp_grad_list,Prop_grad_list,Rep_grad_list,Caus_grad_list = {},{},{},{}
    local list_errors,list_MI, list_corr={},{},{}

    local Prop=Have_Todo(Prior_Used,'Prop') --rename applies_prior()
    local Temp=Have_Todo(Prior_Used,'Temp')
    local Rep=Have_Todo(Prior_Used,'Rep')
    local Caus=Have_Todo(Prior_Used,'Caus')
    print(Prop)
    print(Temp)
    print(Rep)
    print(Caus)

    local coef_Temp=1
    local coef_Prop=1
    local coef_Rep=1
    local coef_Caus=1
    local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}

    print(NB_SEQUENCES..' : sequences. '..NB_BATCHES..' batches')

    for epoch=1, NB_EPOCHS do
       print('--------------Epoch : '..epoch..' ---------------')
       local Temp_loss,Prop_loss,Rep_loss,Caus_loss=0,0,0,0
       local Grad_Temp,Grad_Prop,Grad_Rep,Grad_Caus=0,0,0,0

       xlua.progress(0, NB_BATCHES)
       for numBatch=1, NB_BATCHES do
          indice1=torch.random(1,NB_SEQUENCES-1)
          indice2=torch.random(1,NB_SEQUENCES-1)
          ------------- only one list used----------
          --       print([[====================================================
          -- WARNING TESTING PRIOR, THIS IS NOT RANDOM AT ALL
          -- ====================================================]])
          --       local indice1=8
          --       local indice2=3

          local data1 = load_seq_by_id(indice1)
          local data2 = load_seq_by_id(indice2)

          assert(data1, "Something went wrong while loading data1")
          assert(data2, "Something went wrong while loading data2")

          if Temp then
             Loss,Grad=Rico_Training(Models,'Temp',data1,data2,TEMP_criterion, coef_Temp,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Temp=Grad_Temp+Grad
             Temp_loss=Temp_loss+Loss
          end
          if Prop then
             Loss,Grad=Rico_Training(Models,'Prop',data1,data2, PROP_criterion, coef_Prop,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Prop=Grad_Prop+Grad
             Prop_loss=Prop_loss+Loss
          end
          if Rep then
             Loss,Grad=Rico_Training(Models,'Rep',data1,data2,REP_criterion, coef_Rep,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Rep=Grad_Rep+Grad
             Rep_loss=Rep_loss+Loss
          end
          if Caus then
             Loss,Grad=Rico_Training(Models,'Caus',data1,data2,CAUS_criterion,coef_Caus,LR,BATCH_SIZE, USE_CONTINUOUS)
             Grad_Caus=Grad_Caus+Grad
             Caus_loss=Caus_loss+Loss
          end
          xlua.progress(numBatch, NB_BATCHES)
       end

       local id=name..epoch -- variable used to not mix several log files

       print("Loss Temp", Temp_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Prop", Prop_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Caus", Caus_loss/NB_BATCHES/BATCH_SIZE)
       print("Loss Rep", Rep_loss/NB_BATCHES/BATCH_SIZE)
       print("Saving continuous model in ".. NAME_SAVE..'Continuous')
       if USE_CONTINUOUS then
         model_name = NAME_SAVE..'Continuous'
       else
         model_name = NAME_SAVE
       end
       save_model(Models.Model1, model_name)
    end
end

Tests_Todo={
   {"Prop","Temp","Caus","Rep"}
   --[[
      {"Rep","Caus","Prop"},
      {"Rep","Caus","Temp"},
      {"Rep","Prop","Temp"},
      {"Prop","Caus","Temp"},
      {"Rep","Caus"},
      {"Prop","Caus"},
      {"Temp","Caus"},
      {"Temp","Prop"},
      {"Rep","Prop"},
      {"Rep","Temp"},
      {"Rep"},
      {"Temp"},
      {"Caus"},
      {"Prop"}
   --]]
}



local list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
NB_SEQUENCES= #list_folders_images

for nb_test=1, #Tests_Todo do

   if RELOAD_MODEL then
      Model = torch.load(MODEL_FILE_STRING):double()
   else
      require(MODEL_ARCHITECTURE_FILE)
      Model=getModel(DIMENSION_OUT)
      --graph.dot(Model.fg, 'Our Model')
   end

   if USE_CUDA then
      Model=Model:cuda()
   end

   parameters,gradParameters = Model:getParameters()
   Model2=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   Model3=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   Model4=Model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')

   Models={Model1=Model,Model2=Model2,Model3=Model3,Model4=Model4}

   local Priors=Tests_Todo[nb_test]
   local Log_Folder=Get_Folder_Name(LOG_FOLDER,Priors)
   print("Current test : "..LOG_FOLDER)
   train_Epoch(Models,Priors,Log_Folder,LR, USE_CONTINUOUS)
end

imgs={} --memory is free!!!!!
