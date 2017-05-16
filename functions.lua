require 'const'
---------------------------------------------------------------------------------------
-- Function :save_model(model,path)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function save_model(model,path)
	print("Saved at : "..path)
	if USE_CUDA then
		model:cuda()
  end
	parameters, gradParameters = model:getParameters()
	local lightModel = model:clone():float()
	lightModel:clearState()
	torch.save(path,lightModel)
end

---------------------------------------------------------------------------------------
-- Function : preprocess_image(im, length, width, SpacialNormalization) former preprocess
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function preprocess_image(im, length, width)
	-- Name channels for convenience
	local channels = {'y','u','v'}
	local mean = {}
	local std = {}
	print("preprocess_image im with dataAugmentation...")

	data = torch.Tensor( 3, im:size(2), im:size(3))
	data:copy(im)
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = data[i]:mean()
	   std[i] = data[{i,{},{}}]:std()
	   data[{i,{},{}}]:add(-mean[i])
	   data[{i,{},{}}]:div(std[i])
	end
--[[	local neighborhood = image.gaussian1D(5) -- 5 for face detector training
	-- Define our local normalization operator
	local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1e-4)


	-- Normalize all channels locally:
		-- Normalize all channels locally:
	for c in ipairs(channels) do
		for c in ipairs(channels) do
	      	      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
		      data[{{c},{},{} }] = normalization:forward(data[{{c},{},{} }])
		end
 	end--]]
	--TODO
	--if dataAugmentation then data=dataAugmentationWithNoiseFactor(data, length, width,dataAugmentation) end
	--else
	data=dataAugmentation(data, length, width)
	return data
end

local function gamma(im)
   local Gamma= torch.Tensor(3,3)
   local channels = {'y','u','v'}
   local mean = {}
   local std = {}
   for i,channel in ipairs(channels) do
      for j,channel in ipairs(channels) do
         if i==j then Gamma[i][i] = im[{i,{},{}}]:var()
         else
            chan_i=im[{i,{},{}}]-im[{i,{},{}}]:mean()
            chan_j=im[{j,{},{}}]-im[{j,{},{}}]:mean()
            Gamma[i][j]=(chan_i:t()*chan_j):mean()
         end
      end
   end
   return Gamma
end

local function transformation(im, v,e, fact)
   local transfo=torch.Tensor(3,200,200)
   local Gamma=torch.mv(v,e)
	 if fact~=nil then
		 print("transformation using factor*gamma addition with factor= ")
		 print(fact)
		 for i=1, 3 do
	      transfo[i]=im[i]+Gamma[i]*fact
	   end
	 else
		 for i=1, 3 do
	      transfo[i]=im[i]+Gamma[i]
	   end
	 end
   return transfo
end

function loi_normal(x,y,center_x,center_y,std_x,std_y)
   return math.exp(-(x-center_x)^2/(2*std_x^2))*math.exp(-(y-center_y)^2/(2*std_y^2))
end

function dataAugmentation(im, mean, std)
   local channels = {'r','g','b'}
   local noiseReductionFactor = 4 -- the bigger, less noise
   local length = im:size(2)
   local width = im:size(3)
   local maxShift = 1

   im = normalize(im, mean, std)
   return im

   -- for i=1,3 do
   --    colorShift = torch.uniform(-maxShift,maxShift)
   --    im[{i,{},{}}] = im[{i,{},{}}] + colorShift
   -- end

   -- -- Adding Gaussian noise to the data
   -- noise=torch.rand(3,length,width)/noiseReductionFactor
   -- noise = noise - 0.5/noiseReductionFactor --center noise

   -- im = normalize(im, mean, std):add(noise:float())
   -- return im
end


---------------------------------------------------------------------------------------
-- Function : dataAugmentation(im, length, width) -- TODO
-- Input ():
-- Output ():
-- goal : By using data augmentation we want our network to be more resistant
-- to no task relevant perturbations like luminosity variation or noise
---------------------------------------------------------------------------------------
function dataAugmentationWithNoiseFactor(im, lenght, width,coef_DA)
	local channels = {'y','u','v'}

	gam=gamma(im)
	e, V = torch.eig(gam,'V')
	factors=torch.randn(3)*0.1
	for i=1,3 do e:select(2, 1)[i]=e:select(2, 1)[i]*factors[i] end
	im=transformation(im, V, e:select(2, 1), coef_DA)
	noise=torch.rand(3,lenght,width)
	local mean = {}
	local std = {}
	for i,channel in ipairs(channels) do
	   -- normalize each channel globally:
	   mean[i] = noise[i]:mean()
	   std[i] = noise[{i,{},{}}]:std()
	   noise[{i,{},{}}]:add(-mean[i])
	   noise[{i,{},{}}]:div(std[i])
	end
	--[[
	Gaus=torch.zeros(200,200)
	foyer_x=torch.random(1,200)
	foyer_y=torch.random(1,200)
	std_x=torch.random(1,5)
	std_y=torch.random(1,5)
	for x=1,200 do
		for y=1,200 do
			Gaus[x][y]=loi_normal(x/20,y/20,foyer_x/20,foyer_y/20,std_x,std_y)
		end
	end
	return im+noise--]]
	return im+noise*coef_DA
end

---------------------------------------------------------------------------------------
-- Function :getBatch(imgs, list, indice, length, width, height, Type)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
-- this function search the indice of associated images and take the corresponding images in imgs which are the loaded images of the folder
function getBatch(imgs, list, indice, length, width, height, Type)

   if (indice+1)*length<#list.im1 then
      start=indice*length
   else
      start=#list.im1-length
   end
   if Type=="Prop" then
      Batch=torch.Tensor(4, length,1, width, height)
   else
      Batch=torch.Tensor(2, length,1, width, height)
   end

   for i=1, length do
      Batch[1][i]=imgs[list.im1[start+i]]
      Batch[2][i]=imgs[list.im2[start+i]]
      if Type=="Prop" then
         Batch[3][i]=imgs[list.im3[start+i]]
         Batch[4][i]=imgs[list.im4[start+i]]
      end
   end
   return Batch
end
---------------------------------------------------------------------------------------
-- Function :getRandomBatchFromSeparateList(imgs1, imgs2, txt1, txt2, length, image_width, image_height, Mode, use_simulate_images)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatchFromSeparateList(Data1,Data2, length, Mode)

   local Dim=Data1.images[1]:size()
   if Mode=="Prop" or Mode=="Rep" then
      Batch=torch.Tensor(4, length,Dim[1], Dim[2], Dim[3])
   else
      Batch=torch.Tensor(2, length,Dim[1], Dim[2], Dim[3])
   end

   local im1,im2,im3,im4

   for i=1, length do
      if Mode=="Prop" or Mode=="Rep" then
         Set=get_two_Prop_Pair(Data1.Infos, Data2.Infos)
         im1,im2 = Data1.images[Set.im1], Data1.images[Set.im2]
         im3,im4 = Data2.images[Set.im3], Data2.images[Set.im4]
         Batch[1][i]= im1
         Batch[2][i]= im2
         Batch[3][i]= im3
         Batch[4][i]= im4
      elseif Mode=="Temp" then
         Set=get_one_random_Temp_Set(#Data1.images)
         im1,im2 = Data1.images[Set.im1], Data1.images[Set.im2]
         Batch[1][i]=im1
         Batch[2][i]=im2
      elseif Mode=="Caus" then
         Set=get_one_random_Caus_Set(Data1.Infos, Data2.Infos)

         im1,im2,im3,im4 = Data1.images[Set.im1], Data2.images[Set.im2], Data1.images[Set.im3], Data2.images[Set.im4]
         --The last two are for viz purpose only
         
         Batch[1][i]=im1
         Batch[2][i]=im2
      else
         print "getRandomBatchFromSeparateList Wrong mode "
      end
   end

   --Very useful tool to check if prior are coherent
   if VISUALIZE_IMAGES_TAKEN then
      print("MODE :",Mode)
      visualize_set(im1,im2,im3,im4)
   end

   return Batch

end

function visualize_set(im1,im2,im3,im4)

   if im3 then --Caus or temp
      imgMerge = image.toDisplayTensor({im1,im2,im3,im4})
      image.display{image=imgMerge, win=WINDOW}
   else --Rep or prop
      imgMerge = image.toDisplayTensor({im1,im2})
      image.display{image=imgMerge, win=WINDOW}
   end
   io.read()
end

---------------------------------------------------------------------------------------
-- Function : getRandomBatch(imgs, txt, length, width, height, Mode, use_simulate_images)
-- Input (): Mode: the name of the prior being applied (Prop, Rep, Temp or Caus)
-- Output ():
---------------------------------------------------------------------------------------
function getRandomBatch(Data1, length, Mode)
   --print('getRandomBatch: Data: ')
   --print(Data1)
   --print('getRandomBatch: Data.images size: '..#Data1.images)
   --NOTE we cant do .. Data1.images:size())
   --print(Data1.images)
   --print(Data1.images):size())
   local Dim=Data1.images[1]:size()
   if Mode=="Prop" or Mode=="Rep" then
      Batch=torch.Tensor(4, length,Dim[1], Dim[2], Dim[3])
   else
      Batch=torch.Tensor(2, length,Dim[1], Dim[2], Dim[3])
   end

   for i=1, length do
      if Mode=="Prop" or Mode=="Rep" then
         Set=get_one_random_Prop_Set(Data1.Infos)
         Batch[1][i]=Data1.images[Set.im1]
         Batch[2][i]=Data1.images[Set.im2]
         Batch[3][i]=Data1.images[Set.im3]
         Batch[4][i]=Data1.images[Set.im4]
      elseif Mode=="Temp" then
         Set=get_one_random_Temp_Set(#Data1.images)
         Batch[1][i]=Data1.images[Set.im1]
         Batch[2][i]=Data1.images[Set.im2]
      elseif Mode=="Caus" then
         Set=get_one_random_Caus_Set(Data1.Infos,Data1.Infos)
         Batch[1][i]=Data1.images[Set.im1]
         Batch[2][i]=Data1.images[Set.im2]
      else
         print "getRandomBatch Wrong mode "
      end
   end
   return Batch
end

---------------------------------------------------------------------------------------
-- Function :	Have_Todo(list_prior,prior)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Have_Todo(list_prior,prior)
   local answer=false
   if #list_prior~=0 then
      for i=1, #list_prior do
         if list_prior[i]==prior then answer=true end
      end
   end
   return answer
end

---------------------------------------------------------------------------------------
-- Function :	Get_Folder_Name(Log_Folder,Prior_Used)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folder_Name(Log_Folder,list_prior)
   name=''
   if #list_prior~=0 then
      if #list_prior==1 then
         name=list_prior[1].."_Only"
      elseif #list_prior==4 then
         name='Everything'
      else
         name=list_prior[1]
         for i=2, #list_prior do
            name=name..'_'..list_prior[i]
         end
      end
   end
   return Log_Folder..name..'/'
end



---------------------------------------------------------------------------------------
-- Function :copy_weight(model, AE)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function copy_weight(model, AE)
   model:get(1).weight:copy(AE:get(1).weight)
   model:get(4).weight:copy(AE:get(5).weight)
   return model
end

---------------------------------------------------------------------------------------
-- Function :
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function real_loss(txt,use_simulate_images)

   local REP_criterion=get_Rep_criterion()
   local PROP_criterion=get_Prop_criterion()
   local CAUS_criterion=get_Caus_criterion()
   local TEMP_criterion=nn.MSDCriterion()

   local truth=getTruth(txt,use_simulate_images)

   local temp_loss=0
   local prop_loss=0
   local rep_loss=0
   local caus_loss=0

   local nb_sample=100

   for i=0, nb_sample do
      Set_prop=get_one_random_Prop_Set(txt ,use_simulate_images)
      Set_temp=get_one_random_Temp_Set(#truth)
      Caus_temp=get_one_random_Caus_Set(txt, txt, use_simulate_images)

      joint1=torch.Tensor(1)
      joint2=torch.Tensor(1)
      joint3=torch.Tensor(1)
      joint4=torch.Tensor(1)

      joint1[1]=truth[Caus_temp.im1]
      joint2[1]=truth[Caus_temp.im2]
      caus_loss=caus_loss+CAUS_criterion:updateOutput({joint1, joint2})

      joint1[1]=truth[Set_temp.im1]
      joint2[1]=truth[Set_temp.im2]
      temp_loss=temp_loss+TEMP_criterion:updateOutput({joint1, joint2})

      joint1[1]=truth[Set_prop.im1]
      joint2[1]=truth[Set_prop.im2]
      joint3[1]=truth[Set_prop.im3]
      joint4[1]=truth[Set_prop.im4]
      prop_loss=prop_loss+PROP_criterion:updateOutput({joint1, joint2, joint3, joint4})
      rep_loss=rep_loss+REP_criterion:updateOutput({joint1, joint2, joint3, joint4})
   end

   return temp_loss/nb_sample, prop_loss/nb_sample, rep_loss/nb_sample, caus_loss/nb_sample
end

function load_data(id)

   list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)

   string_preloaded_data = PRELOAD_FOLDER..'preloadedSeq'..id..'.t7'

   if file_exists(string_preloaded_data) then
      data = torch.load(string_preloaded_data)
   else
      local list=images_Paths(list_folders_images[id])
      local txt=list_txt_action[id]
      local txt_reward=list_txt_button[id]
      local txt_state=list_txt_state[id]

      data = load_Part_list(list,txt,txt_reward,IM_LENGTH,IM_HEIGHT,DATA_AUGMENTATION,txt_state)
      torch.save(string_preloaded_data,data)
   end
   return data
end

-- pb si pas de reward....
function getInfos(txt,txt_reward,start,lenght,txt_state)
	local Infos={dx={},dy={},dz={},reward={}}
	local dx=2
	local dy=3
	local dz=4
	local reward_indice=2

local reward=0--!!!new
local tensor_state, label=tensorFromTxt(txt_state)

	local tensor, label=tensorFromTxt(txt)
	local tensor_reward, label=tensorFromTxt(txt_reward)
	local ThereIsReward=false
	for i=start, start+lenght do
		table.insert(Infos.dx,tensor[i][dx])
		table.insert(Infos.dy,tensor[i][dy])
		table.insert(Infos.dz,tensor[i][dz])

if math.floor(tensor_state[i][dx]*100)%20==0 or math.floor(tensor_state[i][dy]*100)%20==0 or math.floor(tensor_state[i][dz]*100)%20==0 then
	ThereIsReward=true
	reward=1
else
	reward=0
 end
table.insert(Infos.reward,reward)
--!!!!!!!!!!!!!!table.insert(Infos.reward,tensor_reward[i][reward_indice])
--print(tensor_reward[i][reward_indice])


--!!!!!!!!!!!!!!if tensor_reward[i][reward_indice]==1 then ThereIsReward=true end
       end

	return Infos,ThereIsReward

end

----
function file_exists(name)
	--tests whether the file can be opened for reading
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

--from Get_Baxter_Files:  loads or createds preload_folder
function loadTrainTest(list_folders_images, crossValStep, PRELOAD_FOLDER)
   imgs = {}
	 print("loadTrainTest: n_data_sequences",PRELOAD_FOLDER)
   preload_name = PRELOAD_FOLDER..'saveImgsRaw.t7'
   if not file_exists(preload_name) then
		  n_data_sequences = #list_folders_images
      print("loadTrainTest: n_data_sequences",n_data_sequences)
      for i=1,n_data_sequences do
         list=images_Paths(list_folders_images[i])
         table.insert(imgs,load_list(list,image_width,image_height,false))
      end
			-- print ('list_folders_images')
			-- print (list_folders_images)
			-- print ('list')
			-- print (list)
      torch.save(preload_name,imgs)
			print("loadTrainTest folder does not exist. Loaded and saved images to "..preload_name)
   else
      imgs = torch.load(preload_name)
			print("loadTrainTest loaded images: "..#imgs.." from "..preload_name)
   end

   -- switch value, because all functions consider the last element to be the test element
   imgs[crossValStep], imgs[#imgs] = imgs[#imgs], imgs[crossValStep]
   print("Preprocess_images_3D... "..#imgs)
   imgs,mean,std = preprocessing(imgs) --preprocess_images_3D(imgs)
	 -- LEARN THAT OPTIONAL PARAMETERS CAN BE OMITTED BY JUST NOT BEING PROVIDED IN LUA
   imgs_test = imgs[#imgs]
   return imgs, imgs_test
end

-- function preprocess_images_3D(imgs)
--    -- Calculate reformat imgs, mean and std for images in train set
--    -- normalize train set and apply to test
--    print('preprocess_images_3D(imgs)')
--    print(#imgs)
--    imgs = scaleAndCrop(imgs)
--    --if not meanStd then
--    mean, std = meanAndStd(imgs)
--   --  else
--   --     mean, std = meanStd[1], meanStd[2]
--   --  end
--
--    numSeq = #imgs-1
--    for i=1,numSeq do
--       for j=1,#(imgs[i]) do
--          im = imgs[i][j]
--          imgs[i][j] =  dataAugmentation(im, mean,std)
--       end
--    end
--    imgs[#imgs] = preprocessingTest(imgs[#imgs], mean,std)
--    return imgs, mean, std
-- end

function preprocessing(imgs,meanStd)
   -- Calculate reformat imgs, mean and std for images in train set
   -- normalize train set and apply to test
   print('preprocessing(imgs,meanStd)')
   print(imgs)
   print(meanStd)
   imgs = scaleAndCrop(imgs)
   if not meanStd then
      mean, std = meanAndStd(imgs)
   else
      mean, std = meanStd[1], meanStd[2]
   end

   numSeq = #imgs-1
   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         im = imgs[i][j]
         imgs[i][j] =  dataAugmentation(im, mean,std)
      end
   end
   imgs[#imgs] = preprocessingTest(imgs[#imgs], mean,std)

   return imgs, mean, std
end

function normalize(im,mean,std)
   for i=1,3 do
      im[{i,{},{}}] = (im[{i,{},{}}]:add(-mean[i])):cdiv(std[i])
   end
   return im
end

-- function load_list(list,length,height, train)
--    local im={}
--    local length=length or 200
--    local height=height or 200
--    for i=1, #list do
--       table.insert(im,getImage(list[i],length,height,train))
-- 	 end
-- end
function load_list(list)
   local im={}
   local lenght=image_width or 200
   local height=image_height or 200
   for i=1, #list do
      table.insert(im,getImage(list[i]))
   end
   return im
end

function scaleAndCrop(imgs, length, height)
   -- Why do i scale and crop after ? Because this is the way it's done under python,
   -- so we need to do the same conversion

   local lengthBeforeCrop = 320
   local lengthAfterCrop = length or 200
   local height = height or 200
   local formatBefore=lengthBeforeCrop.."x"..height

   for s=1,#imgs do
      for i=1,#imgs[s] do
         local img=image.scale(imgs[s][i],formatBefore)
         local img= image.crop(img, 'c', lengthAfterCrop, height)
         imgs[s][i] = img:float()
         -- image.display(img)
         -- io.read()
      end
   end
   return imgs
end

---------------------------------------------------------------------------------------
-- Function : load_list(list,length,height)
-- This method is used by load_data and shouldn't be called on its own
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_Part_list(list,txt,txt_reward,im_length,im_height,data_augmentation,txt_state)

   assert(list, "list not found")
   assert(txt, "Txt not found")
   assert(txt_state, "Txt state not found")
   assert(txt_reward, "Txt reward not found")

   local im={}
   local Infos=getInfos(txt,txt_reward,txt_state)

   for i=1, #(Infos.dx) do
      table.insert(im,getImageAndAugment(list[i],im_length,im_height,data_augmentation))
   end
   return {images=im,Infos=Infos}
end

function getInfos(txt,txt_reward,txt_state)
   local Infos={dx={},dy={},dz={},reward={}}
   local dx=2
   local dy=3
   local dz=4
   local reward_indice=2

   local tensor_state, label=tensorFromTxt(txt_state)

   local tensor, label=tensorFromTxt(txt)
   local tensor_reward, label=tensorFromTxt(txt_reward)
   local there_is_reward=false

   for i=1,tensor_reward:size(1) do
      table.insert(Infos.dx,tensor_state[i][dx])
      table.insert(Infos.dy,tensor_state[i][dy])
      table.insert(Infos.dz,tensor_state[i][dz])

      table.insert(Infos.reward,tensor_reward[i][reward_indice])
      if tensor_reward[i][reward_indice]==1 then there_is_reward=true end
      --print(tensor_reward[i][reward_indice])
   end
   assert(there_is_reward,"Reward is needed in a sequence...")
   return Infos
end

function meanAndStd(imgs)
   local length,height = imgs[1][1][1]:size(1), imgs[1][1][1]:size(2)

   local mean = {torch.zeros(length,height),torch.zeros(length,height),torch.zeros(length,height)}
   local std = {torch.zeros(length,height),torch.zeros(length,height),torch.zeros(length,height)}

   for i=1,3 do
      mean[i] = mean[i]:float()
      std[i] = std[i]:float()
   end

   local numSeq = #imgs-1
   local totImg = 0

   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         mean[1] = mean[1]:add(imgs[i][j][{1,{},{}}]:float())
         mean[2] = mean[2]:add(imgs[i][j][{2,{},{}}]:float())
         mean[3] = mean[3]:add(imgs[i][j][{3,{},{}}]:float())
         totImg = totImg+1
      end
   end

   mean[1] = mean[1] / totImg
   mean[2] = mean[2] / totImg
   mean[3] = mean[3] / totImg

   for i=1,numSeq do
      for j=1,#(imgs[i]) do
         std[1] = std[1]:add(torch.pow(imgs[i][j][{1,{},{}}]:float() - mean[1],2))
         std[2] = std[2]:add(torch.pow(imgs[i][j][{2,{},{}}]:float() - mean[2],2))
         std[3] = std[3]:add(torch.pow(imgs[i][j][{3,{},{}}]:float() - mean[3],2))
      end
   end

   std[1] = torch.sqrt(std[1] / totImg)
   std[2] = torch.sqrt(std[2] / totImg)
   std[3] = torch.sqrt(std[3] / totImg)

   torch.save('Log/meanStdImages.t7',{mean,std})
   return mean,std
end

function preprocessingTest(imgs,mean,std)
   --Normalizing all images
   for i=1,#imgs do
      im = imgs[i]
      imgs[i] = normalize(im,mean,std)
   end
   return imgs
end

function getImage(im)
   if im=='' or im==nil then return nil end
   local image1=image.load(im,3,'byte')
   return image1
   -- local format=length.."x"..height
   -- local img1_rsz=image.scale(image1,format)
   -- return img1_rsz:float()
end

function file_exists(name)
   --tests whether the file can be opened for reading
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


function visualize_image_from_seq_id(seq_id,image_id1,image_id2)
   local data = load_data(seq_id).images
   local image1

   if image_id2 then
      image1 = data[image_id1]
      local image2 = data[image_id2]
      local imgMerge = image.toDisplayTensor({image1,image2})
      image.display{image=imgMerge,win=WINDOW}
   else
      image1 = data[image_id1]
      image.display{image=image1,win=WINDOW}
   end
end
