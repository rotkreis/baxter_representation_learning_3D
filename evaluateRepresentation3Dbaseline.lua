--[[

  THIS IS A 3D VERSION OF THE 1D VERSION IN testRepresentations.lua
 BASELINE
]]--

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'optim'
require 'xlua'   -- xlua provides useful tools, like progress bars
require 'nngraph'
require 'image'
--require 'Get_Baxter_Files'
require 'Get_Images_Set' -- for images_Paths(Path) Get_HeadCamera_View_Files
require 'functions'
require 'const'
require 'printing' --for show_figure
require 'priors'--for get_Rep_criterion
require 'lfs'

require 'math'
require 'string'
require 'MSDC'



-----------------SETTINGS
USE_CUDA = false
nb_part = 50
if not USE_CUDA then
	--	If there is RAM memory problems, one can try to split the dataset in more parts in order to load less image into RAM at one time.
	--  by making "nb_part" larger than 50: -- ToDo: find a value less than 80 and more than 50 for data_baxter_short_seqs and <100 for data_baxter?
	nb_part= 60
	MODEL_ARCHITECTURE_FILE ='./models/minimalNetModel' -- TODO update model_file='./models/topTripleFM_Split'
	--BATCH_SIZE= 1
else
	MODEL_ARCHITECTURE_FILE ='./models/topTripleFM_Split'
	--BATCH_SIZE = 2 --60
end

--MODEL
--MODEL_FILE = MODEL_PATH..MODEL_FILE
require(MODEL_ARCHITECTURE_FILE)
Model = getModel()

---------------------------------------
--MODEL_NAME, name = 'Save97Win/reprLearner1d.t7', '97'
--MODEL_NAME,name = 'reprLearner1dWORKS.t7', 'works'
MODEL_NAME, representationsName = 'reprLearner3d.t7', 'default'  --TODO create
-- if this doesn't exist, it means you didn't run 'train.lua'
MODEL_FULL_PATH = MODEL_PATH..MODEL_NAME
DATA = PRELOAD_FOLDER..'imgsCv1.t7'
PLOT = true
LOADING = false --true

print('Running main script with USE_CUDA flag: '..tostring(USE_CUDA))
print('nb_parts per batch: '..nb_part.." LearningRate: "..LR.." BatchSize: "..BATCH_SIZE..". Using data folder: "..DATA_FOLDER.." Model file Torch: "..MODEL_ARCHITECTURE_FILE..'Preloaded DATA: '..DATA)

local function getReprFromImgs(imgs, PRELOAD_FOLDER, epresentations_name, model_full_path)
  -- we save all metrics that are going to be used in the network for
  -- efficiency (images that fulfill the criteria for each prior and their stats
  -- such as mean and std to avoid multiple computations )
   local fileName = PRELOAD_FOLDER..'allReprSaved'..representations_name..'.t7'

   if file_exists(fileName) then
      return torch.load(fileName)
   else
      print('Preloaded model does not exist: '..fileName..' Run train.lua first! ')
      os.exit()
   end
   print("Calculating all 3D representations with the model: "..fileName)
   print("Number of sequences to calculate :"..#imgs..' totalBatch: '..totalBatch)

   X = {}
   print('getReprFromImgs by loading model: '..MODEL_PATH..MODEL_NAME)
   local model = torch.load(model_full_path)
   for numSeq,seq in ipairs(imgs) do
      print("numSeq",numSeq)
      for i,img in ipairs(seq) do
         x = nn.utils.addSingletonDimension(img)
         X[#X+1] = model:forward(x)[1]
      end
   end
   Xtemp = torch.Tensor(X)
   X = torch.zeros(#X,1)
   X[{{},1}] = Xtemp
   torch.save(fileName,X)
   return X
end

local function HeadPosFromTxts(txts, isData)
   --Since i use this function for creating X tensor for debugging
   -- or y tensor, the label tensor, i need a flag just to tell if i need X or y
   --isData = true => X tensor      isData = false => y tensor
   T = {}
   for l, txt in ipairs(txts) do
      truth = getTruth(txt)
      for i, head_pos in ipairs(truth) do
         T[#T+1] = head_pos
      end
   end
   T = torch.Tensor(T)

   if isData then --is it X or y that you need ?
      Ttemp = torch.zeros(T:size(1),1)
      Ttemp[{{},1}] = T
      T = Ttemp
   end
   return T
end

local function RewardsFromTxts(txts)
   y = {}
   if TASK==2 then
      for l, txt in ipairs(txts) do
         truth = getTruth(txt)
         for i, head_pos in ipairs(truth) do
            if head_pos < 0.1 and head_pos > -0.1 then
               y[#y+1] = 1
            else
               y[#y+1] = 2
            end
         end
      end
   end
   return torch.Tensor(y)
end

local function RandomBatch(X,y,sizeBatch)
   local numSeq = X:size(1)
   batch = torch.zeros(sizeBatch,1)
   y_temp = torch.zeros(sizeBatch)

   for i=1,sizeBatch do
      local id=torch.random(1,numSeq)
      batch[{i,1}] = X[{id,1}]
      y_temp[i] = y[id]
   end
   -- print("batch",batch)
   -- print("y_temp",y_temp)
   -- io.read()
   if USE_CUDA then
     batch = batch:cuda()
     y_temp = y_temp:cuda()
   end
   return batch, y_temp
end

function Rico_Training(model,batch,y,reconstruct, LR)
   local criterion
   local optimizer = optim.adam
   if reconstruct then
      if USE_CUDA then
        criterion = nn.SmoothL1Criterion():cuda()
      else
        criterion = nn.SmoothL1Criterion()
      end
   else
     if USE_CUDA then
      criterion = nn.CrossEntropyCriterion():cuda()
     else
      criterion = nn.CrossEntropyCriterion()
     end
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
      local yhat = model:forward(batch)
      local loss = criterion:forward(yhat,y)
      local grad = criterion:backward(yhat,y)
      model:backward(batch, grad)

      return loss,gradParameters
   end
   optimState={learningRate=LR}
   parameters, loss=optimizer(feval, parameters, optimState)
   return loss[1]
end

function accuracy(X_test,y_test,model)
   local acc = 0
   if USE_CUDA then
    local yhat = model:forward(X_test:cuda())
   else
    local yhat = model:forward(X_test)
  end

   _,yId = torch.max(yhat,2)
   for i=1,X_test:size(1) do
      if yId[i][1]==y_test[i] then
         acc = acc + 1
      end
   end
   return acc/y_test:size(1)
end

function accuracy_reconstruction(X_test,y_test, model)
   local acc = 0
   if USE_CUDA then
    local yhat = model:forward(X_test:cuda())
   else
    local yhat = model:forward(X_test)
   end
   -- print("yhat",yhat[1][1],yhat[2][1],yhat[3][1],yhat[4][1],yhat[60][1])
   -- print("y",truth[1],truth[2],truth[3],truth[4],truth[60])

   for i=1,X_test:size(1) do
      acc = acc + math.sqrt(math.pow(yhat[i][1]-y_test[i],2))
   end
   return acc/X_test:size(1)
end

function rand_accuracy(y_test)
   count = 0
   for i=1,y_test:size(1) do
      if y_test[i]==2 then
         count = count + 1
      end
   end
   return count/y_test:size(1)
end

function createModelReward()
   net = nn.Sequential()
   net:add(nn.Linear(1,3))
   net:add(nn.Tanh())
   net:add(nn.Linear(3,2))
   if USE_CUDA then
    return net:cuda()
   else
    return net
   end
end

function createModelReconstruction()
   net = nn.Sequential()
   net:add(nn.Linear(1,1))
   if USE_CUDA then
    return net:cuda()
   else
    return net
   end
end

function train(X,y, reconstruct)
   reconstruct = reconstruct or true

   local nbList = 10
   local numEx = X:size(1)
   local splitTrainTest = 0.75

   local sizeTest = math.floor(numEx/nbList)

   id_test = {{math.floor(numEx*splitTrainTest), numEx}}
   X_test = X[id_test]
   y_test = y[id_test]

   id_train = {{1,math.floor(numEx*splitTrainTest)}}
   X_train = X[id_train]
   y_train = y[id_train]

   if reconstruct then
      model = createModelReconstruction()
      print("Test accuracy before training",accuracy_reconstruction(X_test,y_test,model))

   else
      model = createModelReward()
      print("Test accuracy before training",accuracy(X_test,y_test,model))
      print("Random accuracy", rand_accuracy(y_test))
   end
   parameters,gradParameters = model:getParameters()

   for epoch=1, NB_EPOCH do
      local lossTemp=0
      for numBatch=1, NB_BATCH do
         batch_temp, y = RandomBatch(X_train,y_train,SIZE_BATCH)
         lossTemp = lossTemp + Rico_Training(model,batch_temp,y, reconstruct, LR)
      end

      if epoch==NB_EPOCH then
         print("lossTemp",lossTemp/NB_BATCH)

         if reconstruct then
            print("Test accuracy = ",accuracy_reconstruction(X_test,y_test,model))
         else
            print("Test accuracy = ",accuracy(X_test,y_test,model))
         end
      end
   end
end

function createPreloadedDataFolder(list_folders_images,list_txt,LOG_FOLDER,use_simulate_images,LR, model_full_path)
   local BatchSize=16
   local nbEpoch=2
   local totalBatch=20
   --local name_save=LOG_FOLDER..'reprLearner1d.t7'
   local coef_Temp=1
   local coef_Prop=1
   local coef_Rep=1
   local coef_Caus=2
   local coef_list={coef_Temp,coef_Prop,coef_Rep,coef_Caus}
   local list_corr={}

   local plot = true
   local loading = true

	--  print('list_folders_images')
	--  print(list_folders_images)
   nbList= #list_folders_images
   local part = 1 --
   local next_part_start_index = part
   for crossValStep=1,nbList do
      models = createModels(model_full_path)
      currentLogFolder=LOG_FOLDER..'CrossVal'..crossValStep..'/' --*
      current_preload_file = PRELOAD_FOLDER..'imgsCv'..crossValStep..'.t7'

      if file_exists(current_preload_file) and loading then
         print("Preloaded Data Already Exists, Loading...")
         imgs = torch.load(current_preload_file)
         imgs_test = imgs[#imgs]
      else
         print("Preloaded Data Does Not Exists. Loading Training and Test and saving to "..current_preload_file)
         local imgs, imgs_test = loadTrainTest(list_folders_images,crossValStep, PRELOAD_FOLDER)
         torch.save(current_preload_file, imgs)
      end

      -- we use last list as test
      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]
      local txt_test=list_txt[#list_txt]
      local truth, next_part_start_index = get_Truth_3D(txt_test,nb_part, next_part_start_index)-- getTruth(txt_test,use_simulate_images)
      print (txt_test)
      print(list_txt)

      assert(#imgs_test==#truth,"Different number of images and corresponding ground truth, something is wrong \nNumber of Images : "..#imgs_test.." and Number of truth values : "..#truth)

      if plot then
         show_figure(truth,currentLogFolder..'GroundTruth.log')
      end
      -- corr=Print_performance(models, imgs_test,txt_test,"First_Test",currentLogFolder,truth,false)
      -- print("Correlation before training : ", corr)
      -- table.insert(list_corr,corr)
      print("Training")
      for epoch=1, nbEpoch do
         print('--------------Epoch : '..epoch..' ---------------')
         local lossTemp=0
         local lossRep=0
         local lossProp=0
         local lossCaus=0

         local causAdded = 0

         for numBatch=1,totalBatch do
            indice1=torch.random(1,nbList-1)
            repeat indice2=torch.random(1,nbList-1) until (indice1 ~= indice2)

            txt1=list_txt[indice1]
            txt2=list_txt[indice2]

            imgs1=imgs[indice1]
            imgs2=imgs[indice2]

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Temp")
            lossTemp = lossTemp + Rico_Training(models,'Temp',batch, coef_Temp,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Caus")
            lossCaus = lossCaus + Rico_Training(models, 'Caus',batch, 1,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Prop")
            lossProp = lossProp + Rico_Training(models, 'Prop',batch, coef_Prop,LR)

            batch=getRandomBatch(imgs1,imgs2,txt1,txt2,BatchSize,"Rep")
            lossRep = lossRep + Rico_Training(models,'Rep',batch, coef_Rep,LR)

            xlua.progress(numBatch, totalBatch)

         end
         --corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,false)
         print("lossTemp",lossTemp/totalBatch)
         print("lossProp",lossProp/totalBatch)
         print("lossRep",lossRep/totalBatch)
         print("lossCaus",lossCaus/(totalBatch+causAdded))
      end
      corr=Print_performance(models, imgs_test,txt_test,"Test",currentLogFolder,truth,plot)
      --show_figure(list_corr,currentLogFolder..'correlation.log','-')

      --for reiforcement, we need mean and std to normalize representation
      print("SAVING MODEL AND REPRESENTATIONS")
      saveMeanAndStdRepr(imgs)
      models.model1:float()
      --save_model(models.model1,name_save)
      list_txt[crossValStep],list_txt[#list_txt] = list_txt[#list_txt], list_txt[crossValStep]
   end
end

--from functions 1D
function createModels(MODEL_FULL_PATH)
   if LOADING then
      print("Loading Model..."..MODEL_FULL_PATH)
      if file_exists(MODEL_FULL_PATH) then
         model = torch.load(MODEL_FULL_PATH)  --LOG_FOLDER..'20e.t7'
      else
         print("Model file does not exist!")
         os.exit()
      end
   else
      model=getModel()
      print(model)
   end

   model=model:cuda()
   parameters,gradParameters = model:getParameters()
   model2=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model3=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   model4=model:clone('weight','bias','gradWeight','gradBias','running_mean','running_std')
   models={model1=model,model2=model2,model3=model3,model4=model4}
   return models
end


local function getHandPosFromTxts(txts, nb_part, part)
   --Since i use this function for creating X tensor for debugging
   -- or y tensor, the label tensor, i need a flag just to tell if i need X or y
   --isData = true => X tensor      isData = false => y tensor
   T = {}
   for l, txt in ipairs(txts) do
      truth, part = get_Truth_3D(txt, nb_part, part)
      for i, hand_pos in ipairs(truth) do
         T[#T+1] = hand_pos
      end
   end

   T = torch.Tensor(T)

   if isData then --is it X or y that you need ?
      Ttemp = torch.zeros(T:size(1),1)
      Ttemp[{{},1}] = T
      T = Ttemp
   end

   return T
end

local function getRewardsFromTxts(txt_joint, nb_parts, part)
   y = {}
    for l, txt in ipairs(txt_joint) do
       truth, part = get_Truth_3D(txt_joint, nb_part, part)
       for i, head_pos in ipairs(truth) do
          if head_pos.x < 0.1 and head_pos.x > -0.1 then
						--TODO: get real positions of the button
             y[#y+1] = 1  -- negative vs positive reward
          else
             y[#y+1] = 2
          end
       end
    end
   return torch.Tensor(y)
end


------------------------------------
-- Our representation learnt should be coordinate independent, as it is not aware of
-- what is x,y,z and thus, we should be able to reconstruct the state by switching
-- to y,x,z, or if we add Gaussian noise to the true positions x,y,z of Baxter arm.
-- These will be our score baseline to compare with
reconstructingTask = true
local dataAugmentation=true
--local list_folders_images, list_txt=Get_HeadCamera_HeadMvt() local _, list_txt=Get_HeadCamera_HeadMvt(DATA_FOLDER)

if not file_exists(PRELOAD_FOLDER) then
   lfs.mkdir(PRELOAD_FOLDER)
end
if not file_exists(LOG_FOLDER) then
   lfs.mkdir(LOG_FOLDER)
end
---
indice_test = 1--nbList --4 --nbList
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)
print("Got list_folders_images: ")
print(#list_folders_images)
if #list_folders_images >0 then
	--print('get_images_paths'..list_folders_images[indice_test])
	local list_image_paths= get_images_paths(list_folders_images[indice_test])
	txt_test=list_txt_state[indice_test]
	txt_reward_test=list_txt_button[indice_test]
	part_test=1
	Data_test=load_Part_list(list_image_paths,txt_test,txt_reward_test,image_width,image_height,nb_part,part_test,0,txt_test)
	local truth=get_Truth_3D(txt_test,nb_part,part_test) -- 100 DoubleTensor of size 3
	print("Plotting the truth... ")
	--show_figure(truth, LOG_FOLDER..'The_Truth.Log','Truth',Data_test.Infos)
	print("Computing performance... ")
	--Print_performance(Models, Data_test,txt_test,txt_reward_test,"First_Test",LOG_FOLDER,truth)
	---

	createPreloadedDataFolder(list_folders_images,list_txt,LOG_FOLDER,use_simulate_images,LR,MODEL_FULL_PATH)
	imgs={}
	------------------------------------
	local imgs = torch.load(DATA)
	imgs[1], imgs[#imgs] = imgs[#imgs], imgs[1] -- Because during database creation we swapped those values


	if reconstructingTask then
	   y = getHandPosFromTxts(list_txt, nb_part, 1)
	else
	   y = getHandPosRewardsFromTxts(list_txt, nb_part, 1)
	end

	--X = HeadPosFromTxts(list_txt,true)
	-- predict:
	X = getReprFromImgs(imgs, PATH_PRELOAD_DATA,representationsName, MODEL_FULL_PATH)

	NB_BATCH=math.floor(X:size(1)/SIZE_BATCH)

	train(X,y,reconstructingTask)
else
	print("Input image files not found, check your DATA_FOLDER global variable (should be named 'simpleData3D')")
	os.exit()
end
