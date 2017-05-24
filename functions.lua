require 'const'
require 'image'
require 'Get_Images_Set'
---------------------------------------------------------------------------------------
-- Function :save_model(model,path)
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function save_model(model)

   path = LOG_FOLDER..NAME_SAVE
   lfs.mkdir(path)
   file_string = path..'/'..NAME_SAVE..'.t7'

   os.execute("cp hyperparams.lua "..path)

   print("Saved at : "..path)
   torch.save(file_string,model)

   f = io.open('lastModel.txt','w')
   f:write(path..'\n'..NAME_SAVE..'.t7')
   f:close()
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

         im2,im3 = im3,im2 --I switch them for a better viz

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

function getRandomBatchFromSeparateListContinuous(Data1,Data2, batchSize, Mode)
   local Dim=Data1.images[1]:size()
   if Mode=="Prop" or Mode=="Rep" then
      Batch=torch.Tensor(4, batchSize,Dim[1], Dim[2], Dim[3])
   else
      Batch=torch.Tensor(2, batchSize,Dim[1], Dim[2], Dim[3])
   end

   local im1,im2,im3,im4

   for i=1, batchSize do
      if Mode=="Prop" or Mode=="Rep" then
         Set =get_two_Prop_Pair_and_actions(Data1.Infos, Data2.Infos)
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
         Set =get_one_random_Caus_Set_and_actions(Data1.Infos, Data2.Infos)

         im1,im2,im3,im4 = Data1.images[Set.im1], Data2.images[Set.im2], Data1.images[Set.im3], Data2.images[Set.im4]
         --The last two are for viz purpose only

         Batch[1][i]=im1
         Batch[2][i]=im2
      else
         print "getRandomBatchFromSeparateListContinuous Wrong mode "
      end
   end

   --Very useful tool to check if prior are coherent
   if VISUALIZE_IMAGES_TAKEN then
      print("MODE :",Mode)
      visualize_set(im1,im2,im3,im4)
   end
   return Batch, Set.act1, Set.act2
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
-- Function :
-- Input ():
-- Output (): TODO: REMOVE
---------------------------------------------------------------------------------------
-- function real_loss(txt,use_simulate_images)
--    local REP_criterion=get_Rep_criterion()
--    local PROP_criterion=get_Prop_criterion()
--    local CAUS_criterion=get_Caus_criterion()
--    local TEMP_criterion=nn.MSDCriterion()
--
--    local truth=getTruth(txt,use_simulate_images)
--
--    local temp_loss=0
--    local prop_loss=0
--    local rep_loss=0
--    local caus_loss=0
--
--    local nb_sample=100
--
--    for i=0, nb_sample do
--       Set_prop=get_one_random_Prop_Set(txt ,use_simulate_images)
--       Set_temp=get_one_random_Temp_Set(#truth)
--       Caus_temp=get_one_random_Caus_Set(txt, txt, use_simulate_images)
--
--       joint1=torch.Tensor(1)
--       joint2=torch.Tensor(1)
--       joint3=torch.Tensor(1)
--       joint4=torch.Tensor(1)
--
--       joint1[1]=truth[Caus_temp.im1]
--       joint2[1]=truth[Caus_temp.im2]
--       caus_loss=caus_loss+CAUS_criterion:updateOutput({joint1, joint2})
--
--       joint1[1]=truth[Set_temp.im1]
--       joint2[1]=truth[Set_temp.im2]
--       temp_loss=temp_loss+TEMP_criterion:updateOutput({joint1, joint2})
--
--       joint1[1]=truth[Set_prop.im1]
--       joint2[1]=truth[Set_prop.im2]
--       joint3[1]=truth[Set_prop.im3]
--       joint4[1]=truth[Set_prop.im4]
--       prop_loss=prop_loss+PROP_criterion:updateOutput({joint1, joint2, joint3, joint4})
--       rep_loss=rep_loss+REP_criterion:updateOutput({joint1, joint2, joint3, joint4})
--    end
--
--    return temp_loss/nb_sample, prop_loss/nb_sample, rep_loss/nb_sample, caus_loss/nb_sample
-- end


function load_seq_by_id(id)
   local string_preloaded_and_normalized_data = PRELOAD_FOLDER..'preloaded_'..DATA_FOLDER..'_Seq'..id..'_normalized.t7'

   -- DATA + NORMALIZATION EXISTS
   if file_exists(string_preloaded_and_normalized_data) then
      data = torch.load(string_preloaded_and_normalized_data)
   else   -- DATA DOESN'T EXIST AT ALL
      list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(DATA_FOLDER)

      -- print("list_folders_images",list_folders_images)
      -- print("list_folders_images",list_txt_action)
      -- print("list_txt_button",list_txt_button)
      -- print("list_txt_state",list_txt_state)


      local list=images_Paths(list_folders_images[id])
      local txt=list_txt_action[id]
      local txt_reward=list_txt_button[id]
      local txt_state=list_txt_state[id]

      data = load_Part_list(list,txt,txt_reward,txt_state)
      torch.save(string_preloaded_and_normalized_data,data)
   end
   return data
end

function scaleAndCrop(img)
   -- Why do i scale and crop after ? Because this is the way it's done under python,
   -- so we need to do the same conversion

   -- local lengthBeforeCrop = 320 --Tuned by hand, that way, when you scale then crop, the image is 200x200

   -- local lengthAfterCrop = IM_LENGTH
   -- local height = IM_HEIGHT
   -- local formatBefore=lengthBeforeCrop.."x"..height

   local format=IM_LENGTH.."x"..IM_HEIGHT
   local imgAfter=image.scale(img,format)

   if VISUALIZE_IMAGE_CROP then
      dim1_before = img:size(1)
      dim2_before = img:size(2)
      dim3_before = img:size(3)

      dim1_after = imgAfter:size(1)
      dim2_after = imgAfter:size(2)
      dim3_after = imgAfter:size(3)

      imgAfterPadded =torch.zeros(dim1_before,dim2_before, dim3_before)
      imgAfterPadded[{{1,dim1_after},{1,dim2_after},{1,dim3_after}}] =
         imgAfter

      local imgMerge = image.toDisplayTensor({img,imgAfterPadded})
      print("Before and After scale")
      image.display{image=imgMerge,win=WINDOW}
      io.read()
   end

   return imgAfter
end

---------------------------------------------------------------------------------------
-- Function : load_list(list,length,height)
-- This method is used by load_data and shouldn't be called on its own
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function load_Part_list(list,txt,txt_reward,txt_state)

   assert(list, "list not found")
   assert(txt, "Txt not found")
   assert(txt_state, "Txt state not found")
   assert(txt_reward, "Txt reward not found")

   local im={}
   local Infos=getInfos(txt,txt_reward,txt_state)

   for i=1, #(Infos[1]) do
      table.insert(im,getImageFormated(list[i]))
   end

   return {images=im,Infos=Infos}
end

function is_out_of_bound(list_pos)

   -- For each dimension you check if the value is inside
   -- barrier fix by MIN_TABLE and MAX_TABLE
   for dim=1,#list_pos do
      if list_pos[dim] < MIN_TABLE[dim] or list_pos[dim] > MAX_TABLE[dim] then
         return true
      end
   end
   return false
end

function getInfos(txt,txt_reward,txt_state)

   local Infos={}
   for dim=1,DIMENSION_IN do
      Infos[dim] = {}
   end
   Infos.reward = {}

   local reward_indice=REWARD_INDICE

   local tensor_state, label=tensorFromTxt(txt_state)

   local tensor, label=tensorFromTxt(txt)
   local tensor_reward, label=tensorFromTxt(txt_reward)
   local there_is_reward=false

   for i=1,tensor_reward:size(1) do

      local last_pos = {}

      for dim=1,#INDICE_TABLE do
         id_of_dim_in_tensor = INDICE_TABLE[dim]
         local value = tensor_state[i][id_of_dim_in_tensor]
         table.insert(Infos[dim],value)
         table.insert(last_pos, value) -- For out_of_bound func
      end

      local reward = tensor_reward[i][reward_indice]

      if reward~=0 then
         there_is_reward=true
         table.insert(Infos.reward,reward)
      elseif is_out_of_bound(last_pos) then
         there_is_reward=true
         table.insert(Infos.reward,-1)
      else
         table.insert(Infos.reward,0)
      end
      --print(tensor_reward[i][reward_indice])
   end
   assert(there_is_reward,"Reward is needed in a sequence...")
   return Infos
end

function calculate_mean_and_std()
   -- This function can work on its own
   -- Just need the global variable DATA_FOLDER to be set

   print("Calculating Mean and Std for all images in ", DATA_FOLDER)

   local imagesFolder = DATA_FOLDER

   local mean = {torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT)}
   local std = {torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT),torch.zeros(IM_LENGTH,IM_HEIGHT)}
   local totImg = 0

   for i=1,3 do
      mean[i] = mean[i]:float()
      std[i] = std[i]:float()
   end

   for seqStr in lfs.dir(imagesFolder) do
      if string.find(seqStr,'record') then
         print("seqStr",seqStr)
         local imagesPath = imagesFolder..'/'..seqStr..'/'..SUB_DIR_IMAGE
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'jpg') then
               totImg = totImg + 1
               local fullImagesPath = imagesPath..'/'..imageStr
               local img=image.load(fullImagesPath,3,'float')
               img = scaleAndCrop(img)

               mean[1] = mean[1]:add(img[{1,{},{}}])
               mean[2] = mean[2]:add(img[{2,{},{}}])
               mean[3] = mean[3]:add(img[{3,{},{}}])
            end
         end
      end
   end

   mean[1] = mean[1] / totImg
   mean[2] = mean[2] / totImg
   mean[3] = mean[3] / totImg

   for seqStr in lfs.dir(imagesFolder) do
      if string.find(seqStr,'record') then
         local imagesPath = imagesFolder..'/'..seqStr..'/'..SUB_DIR_IMAGE
         for imageStr in lfs.dir(imagesPath) do
            if string.find(imageStr,'jpg') then
               local fullImagesPath = imagesPath..'/'..imageStr
               local img=image.load(fullImagesPath,3,'float')
               img = scaleAndCrop(img)
               std[1] = std[1]:add(torch.pow(img[{1,{},{}}]-mean[1],2))
               std[2] = std[2]:add(torch.pow(img[{2,{},{}}]-mean[2],2))
               std[3] = std[3]:add(torch.pow(img[{3,{},{}}]-mean[3],2))
            end
         end
      end
   end

   std[1] = torch.sqrt(std[1] / totImg)
   std[2] = torch.sqrt(std[2] / totImg)
   std[3] = torch.sqrt(std[3] / totImg)

   im_mean = torch.zeros(3,200,200)
   im_std = torch.zeros(3,200,200)

   for i=1,3 do
      im_mean[i] = mean[i]
      im_std[i] = std[i]
   end

   im_mean = im_mean:float()
   im_std = im_std:float()
   torch.save(STRING_MEAN_AND_STD_FILE,{mean=im_mean,std=im_std})
   return im_mean,im_std
end


function normalize(im)

   local meanStd, mean, std, im_norm, imgMerge

   -- print("im1")
   -- image.display{image=im, win=WINDOW}
   -- io.read()

   if file_exists(STRING_MEAN_AND_STD_FILE) then
      meanStd = torch.load(STRING_MEAN_AND_STD_FILE)
      mean = meanStd.mean
      std = meanStd.std
   else
      mean, std = calculate_mean_and_std()
   end

   im_norm = torch.add(im,-mean)
   --im_norm = torch.cdiv(im_norm, std)

   -- print("im2",im[1][1][1])
   -- image.display{image=im, win=WINDOW}
   -- io.read()

   if VISUALIZE_MEAN_STD then
      --imgMerge = image.toDisplayTensor({mean,std,im,im_norm})
      imgMerge = image.toDisplayTensor({mean,im,im_norm})

      print("Mean, im, im_norm")
      image.display{image=imgMerge, win=WINDOW}
      io.read()
   end

   return im_norm
end

function getImageFormated(im)
   if im=='' or im==nil then error("im is nil, this is not an image") end
   local img=image.load(im,3,'float')
   img = scaleAndCrop(img)
   if NORMALIZE_IMAGE then
      img = normalize(img)
   end
   return img
end


function file_exists(name)
   --tests whether the file can be opened for reading
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end


function visualize_image_from_seq_id(seq_id,image_id1,image_id2, another_window)
   local data = load_seq_by_id(seq_id).images
   local image1

   if image_id2 then
      image1 = data[image_id1]
      local image2 = data[image_id2]
      local imgMerge = image.toDisplayTensor({image1,image2})

      if another_window then
         image.display{image=imgMerge,win=a}
      else
         image.display{image=imgMerge,win=WINDOW}
      end

   else
      image1 = data[image_id1]
      image.display{image=image1,win=WINDOW}
   end
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
