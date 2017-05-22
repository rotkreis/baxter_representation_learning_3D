require 'const'

CLOSE_ENOUGH_PRECISION_THRESHOLD = 0.6
ACTION_AMPLITUDE = 0.01
GAUSSIAN_SIGMA = 0.5

---------------------------------------------------------------------------------------
-- Function : images_Paths(path)  #TODO remove to avoid conflict with 1D
-- Input (Path): path of a Folder which contains jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
function images_Paths(folder_containing_jpgs)
   local listImage={}
   --print('images_Paths: ', folder_containing_jpgs)
   --folder_containing_jpgs="./data_baxter" -- TODO: make it work by passing it as a parameter
   for file in paths.files(folder_containing_jpgs) do
      --print('getting image path:  '..file)
      -- We only load files that match the extension
      if file:find('jpg' .. '$') then
         -- and insert the ones we care about in our table
         table.insert(listImage, paths.concat(folder_containing_jpgs,file))
         --print('Inserted image :  '..paths.concat(folder_containing_jpgs,file))
      end
   end
   table.sort(listImage)
   --print('Loaded images from Path: '..folder_containing_jpgs)
   return listImage
end

---------------------------------------------------------------------------------------
-- Function : images_Paths(path)
-- Input (Path): path of a Folder which contains jpg images
-- Output : list of the jpg files path
---------------------------------------------------------------------------------------
-- function get_images_paths(folder_containing_jpgs)
--    local listImage={}
--    print('get_images_paths: ', folder_containing_jpgs)
--    --folder_containing_jpgs="./data_baxter" -- TODO: make it work by passing it as a parameter
--
--    for file in paths.files(folder_containing_jpgs) do
--       --print('getting image path:  '..file)
--       -- We only load files that match the extension
--       if file:find('jpg' .. '$') then
--          -- and insert the ones we care about in our table
--          table.insert(listImage, paths.concat(folder_containing_jpgs,file))
--          --print('Inserted image :  '..paths.concat(folder_containing_jpgs,file))
--       end
--    end
--    table.sort(listImage) --print('got_images_paths from Path: '..folder_containing_jpgs)
--    print(listImage)
--    return listImage
-- end

---------------------------------------------------------------------------------------
-- Function :
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function txt_path(Path, including)
   local including=including or ""
   local txt=nil
   for file in paths.files(Path) do
      if file:find(including..'.txt' .. '$') then
         txt=paths.concat(Path,file)
      end
   end
   return txt
end

---------------------------------------------------------------------------------------
-- Function :
-- Input ():
-- Output ():
---------------------------------------------------------------------------------------
function Get_Folders(Path, including, excluding,list)
   local list=list or {}
   local incl=including or ""
   local excl=excluding or "uyfouhjbhytfoughl" -- random motif
   for file in paths.files(Path) do
      -- We only load files that match 2016 because we know that there are the folder we are interested in
      if file:find(incl) and (not file:find(excl)) then
         -- and insert the ones we care about in our table
         --print('Get_Folders '..Path..' found search pattern: '..incl..' in filename: '..paths.concat(Path,file))
         table.insert(list, paths.concat(Path,file))
         --  else
         -- 	 print('Get_Folders '..Path..' did not find pattern: '..incl..' Check the structure of your data folders')
      end
   end
   return list
end

---------------------------------------------------------------------------------------
-- Function : Get_HeadCamera_HeadMvt(use_simulate_images) --TODO: Get_HeadCamera_HeadMvt?
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images
-- Output (list_head_left): list of the images directories path
-- Output (list_txt):  txt list associated to each directories (this txt file contains the grundtruth of the robot position)
---------------------------------------------------------------------------------------
function Get_HeadCamera_View_Files(Path)
	local use_simulate_images=use_simulate_images or false
	local Paths=Get_Folders(Path,'record')  --TODO? formerly, for 1D, 'record' param was not needed
	list_folder={}
	list_txt_button={}
	list_txt_action={}
	list_txt_state={}

	for i=1, #Paths do
		list_folder=Get_Folders(Paths[i],'recorded','txt',list_folder)
		table.insert(list_txt_button, txt_path(Paths[i],"is_pressed"))
		table.insert(list_txt_action, txt_path(Paths[i],"endpoint_action"))
		table.insert(list_txt_state, txt_path(Paths[i],"endpoint_state"))
	end
	table.sort(list_txt_button) -- file recorded_button_is_pressed.txt
	table.sort(list_txt_action) --file recorded_robot_limb_left_endpoint_action.txt
	table.sort(list_txt_state)--recroded_robot_libm_left_endpoint_state  -- for the hand position
	table.sort(list_folder) --recorded_cameras_head_camera_2_image_compressed
	return list_folder, list_txt_action,list_txt_button, list_txt_state
end

---------------------------------------------------------------------------------------
-- Function : tensorFromTxt(path)
-- Input (path) : path of a txt file which contain position of the robot
-- Output (torch.Tensor(data)): tensor with all the joint values (col: joint, line : indice)
-- Output (labels):  name of the joint
---------------------------------------------------------------------------------------
function tensorFromTxt(path)
   local data, raw = {}, {}
   local rawCounter, columnCounter = 0, 0
   local nbFields, labels, _line = nil, nil, nil
   --print('tensorFromTxt path:',path)
   for line in io.lines(path)  do   ---reads each line in the .txt data file
      local comment = false
      if line:sub(1,1)=='#' then
         comment = true
         line = line:sub(2)
      end
      rawCounter = rawCounter +1
      columnCounter=0
      raw = {}
      for value in line:gmatch'%S+' do
         columnCounter = columnCounter+1
         raw[columnCounter] = tonumber(value)
      end

      -- we check that every row contains the same number of data
      if rawCounter==1 then
         nbFields = columnCounter
      elseif columnCounter ~= nbFields then
         error("data dimension for " .. rawCounter .. "the sample is not consistent with previous samples'")
      end

      if comment then labels = raw else table.insert(data,raw) end
   end
   return torch.Tensor(data), labels
end

--============== Tools to get action from get state ===========
--=============================================================
function action_amplitude(infos,id1, id2)
   -- Returns delta in coordinate positions from start to end of the action
   local action = {}
   action.x = infos.dx[id1] - infos.dx[id2]
   action.y = infos.dy[id1] - infos.dy[id2]
   action.z = infos.dz[id1] - infos.dz[id2]
   return action
end

function is_same_action(action1,action2)
   if arrondit(action1.x - action2.x)==0 and
      arrondit(action1.y - action2.y)==0 and
   arrondit(action1.z - action2.z)==0 then
      return true
   else
      return false
   end
end

-- Making actions not be the same but close enough for the continous handling of priors
function actions_are_close_enough(action1,action2)
   if arrondit(action1.x - action2.x) < CLOSE_ENOUGH_PRECISION_THRESHOLD and
      arrondit(action1.y - action2.y) < CLOSE_ENOUGH_PRECISION_THRESHOLD and
   arrondit(action1.z - action2.z) < CLOSE_ENOUGH_PRECISION_THRESHOLD then
      return true
   else
      return false
   end
end

function actions_distance(action1, action2)
  return math.sqrt(math.abs(arrondit(action1.x - action2.x)) + math.abs(arrondit(action1.y - action2.y))+ math.abs(arrondit(action1.z - action2.z)))

---------------------------------------------------------------------------------------
-- Function : get_one_random_Temp_Set(list_im)
-- Input (list_lenght) : lenght of the list of images
-- Output : 2 indices of images which are neightboor in the list (and in time)
---------------------------------------------------------------------------------------
function get_one_random_Temp_Set(list_lenght)
   indice=torch.random(1,list_lenght-1)
   return {im1=indice,im2=indice+1}
end

function get_one_random_Prop_Set_and_action_deltas(Infos1)
   return get_two_Prop_Pair_and_action_deltas(Infos1,Infos1)
end

---------------------------------------------------------------------------------------
-- Function : get_two_Prop_Pair_and_action_deltas(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior.
-- The variation of joint for one pair should be close enough (<CLOSE_ENOUGH_PRECISION_THRESHOLD) in continuous actions, to the variation for the second
---------------------------------------------------------------------------------------
function get_two_Prop_Pair_and_action_deltas(Infos1, Infos2)

   local watchDog=0
   local size1=#Infos1.dx
   local size2=#Infos2.dx

   local vector=torch.randperm(size2-1)
   local action_deltas = {}
   while watchDog<100 do
      local id_ref_action_begin=torch.random(1,size1-1)

      if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
         repeat id_ref_action_end=torch.random(1,size1) until (id_ref_action_begin ~= id_ref_action_end)
      else
         id_ref_action_end=id_ref_action_begin+1
      end

      action1 = action_amplitude(Infos1,id_ref_action_begin, id_ref_action_end)

      for i=1, size2-1 do
         local id_second_action_begin=vector[i]

         if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
            for id_second_action_end in ipairs(torch.totable(torch.randperm(size2))) do
               action2 = action_amplitude(Infos2, id_second_action_begin, id_second_action_end)
               if actions_are_close_enough(action1, action2) then --is_same_action(action1, action2) then
                  action_deltas[1] = actions_distance(action1, action2)
                  return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end}, action_deltas
               end
            end
         else --USE THE NEXT IMAGE IN THE SEQUENCE
            id_second_action_end=id_second_action_begin+1
            action2 = action_amplitude(Infos2, id_second_action_begin, id_second_action_end)
            if actions_are_close_enough(action1, action2) then
               action_deltas[1] = actions_distance(action1, action2)
               return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end}, action_deltas
            end
         end
      end
      watchDog=watchDog+1
   end
   error("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

---------------------------------------------------------------------------------------
-- Function : get_one_random_Caus_Set_and_action_deltas(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Training with prop prior,
--  and an array of the delta in between actions (the distance in between 2 actions as Euclidean distance)
-- The variation of joint for one pair should be close enough (<CLOSE_ENOUGH_PRECISION_THRESHOLD) in continuous actions, to the variation for the second
---------------------------------------------------------------------------------------
function get_one_random_Caus_Set_and_action_deltas(Infos1, Infos2)
  local watchDog=0
  local dx=2
  local dy=3
  local dz=4

  local size1=#Infos1.dx
  local size2=#Infos2.dx
  vector=torch.randperm(size2-1)
  local action_deltas = {}

  while watchDog<100 do
     repeat
        id_ref_action_begin= torch.random(1,size2-1)

        if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
           id_ref_action_end  = torch.random(1,size2)
        else
           id_ref_action_end  = id_ref_action_begin+1
        end
     until(Infos2.reward[id_ref_action_begin]==0 and Infos2.reward[id_ref_action_end]==1)

     if VISUALIZE_CAUS_IMAGE then
        visualize_image_from_seq_id(indice2,id_ref_action_begin,id_ref_action_end)
     end

     action1 = action_amplitude(Infos2, id_ref_action_begin, id_ref_action_end)


     if CLAMP_CAUSALITY and not EXTRAPOLATE_ACTION then
        -- WARNING, THIS IS DIRTY, need to do continous prior
        action1.x = clamp_causality_prior_value(action1.x)
        action1.y = clamp_causality_prior_value(action1.y)
        action1.z = clamp_causality_prior_value(action1.z)
     end

     -- print("id1",id_ref_action_begin)
     -- print("id2",id_ref_action_end)

     --print("action1",action1.x,action1.y,action1.z)
     -- io.read()

     for i=1, size1-1 do
        id_second_action_begin=torch.random(1,size1-1)

        if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
           id_second_action_end=torch.random(1,size1)
        else
           id_second_action_end=id_second_action_begin+1
        end

        if Infos1.reward[id_second_action_begin]==0 and Infos1.reward[id_second_action_end]==0 then
           action2 = action_amplitude(Infos1, id_second_action_begin, id_second_action_end)
           --print("action2",action2.x,action2.y,action2.z)

           if actions_are_close_enough(action1, action2) then
              action_deltas[1] = actions_distance(action1, action2)
              return {im1=id_second_action_begin,im2=id_ref_action_begin, im3=id_second_action_end, im4=id_ref_action_end}, action_deltas
           end
        end
     end
     watchDog=watchDog+1
  end
  error("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end


-- I need to search images representing a starting state.
-- then the same action applied to this to state (the same variation of joint) should lead to a different reward.
-- for instance we choose for reward the fact to have a joint = 0

-- NB : the two states will be took in different list but the two list can be the same
function get_one_random_Caus_Set(Infos1,Infos2)
   local watchDog=0
   local dx=2
   local dy=3
   local dz=4

   local size1=#Infos1.dx
   local size2=#Infos2.dx
   vector=torch.randperm(size2-1)

   while watchDog<100 do
      repeat
         id_ref_action_begin= torch.random(1,size2-1)

         if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
            id_ref_action_end  = torch.random(1,size2)
         else
            id_ref_action_end  = id_ref_action_begin+1
         end
      until(Infos2.reward[id_ref_action_begin]==0 and Infos2.reward[id_ref_action_end]==1)

      if VISUALIZE_CAUS_IMAGE then
         visualize_image_from_seq_id(indice2,id_ref_action_begin,id_ref_action_end)
      end

      action1 = action_amplitude(Infos2, id_ref_action_begin, id_ref_action_end)

      if CLAMP_CAUSALITY and not EXTRAPOLATE_ACTION then
         -- WARNING, THIS IS DIRTY, need to do continous prior
         action1.x = clamp_causality_prior_value(action1.x)
         action1.y = clamp_causality_prior_value(action1.y)
         action1.z = clamp_causality_prior_value(action1.z)
      end

      -- print("id1",id_ref_action_begin)
      -- print("id2",id_ref_action_end)

      --print("action1",action1.x,action1.y,action1.z)
      -- io.read()

      for i=1, size1-1 do
         id_second_action_begin=torch.random(1,size1-1)

         if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
            id_second_action_end=torch.random(1,size1)
         else
            id_second_action_end=id_second_action_begin+1
         end

         if Infos1.reward[id_second_action_begin]==0 and Infos1.reward[id_second_action_end]==0 then
            action2 = action_amplitude(Infos1, id_second_action_begin, id_second_action_end)
            --print("action2",action2.x,action2.y,action2.z)

            if is_same_action(action1, action2) then
               return {im1=id_second_action_begin,im2=id_ref_action_begin, im3=id_second_action_end, im4=id_ref_action_end}
            end
         end
      end
      watchDog=watchDog+1
   end
   error("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

---------------------------------------------------------------------------------------
-- Function : arrondit(value)
-- Input (tensor) :
-- Input (head_pan_indice) :
-- Output (tensor):
---------------------------------------------------------------------------------------
function arrondit(value, prec)
   local prec = prec or 0.05--0.05 precision
   divFactor = 1/prec

   floor=math.floor(value*divFactor)/divFactor
   ceil=math.ceil(value*divFactor)/divFactor
   if math.abs(value-ceil)>math.abs(value-floor) then result=floor
   else result=ceil end
   return result
end

function clamp_causality_prior_value(value, prec, action_amplitude)
   -- ======================================================
   -- WARNING THIS VERY DIRTY, WE SHOULD DO CONTINOUS PRIOR
   -- INSTEAD OF THIS
   -- ======================================================
   prec = prec or CLOSE_ENOUGH_PRECISION_THRESHOLD --0.01  --TODO: Use CLOSE_ENOUGH_PRECISION_THRESHOLD
   action_amplitude = action_amplitude or ACTION_AMPLITUDE --0.05 --An action has an amplitude either of
   --- 0 or 0.05 in the 'simple3D' database (on each axis). In continuous data such as baxter_babbling, this is arbitrary threshold

   if math.abs(value) < prec then
      value = 0
   else
      value = sign(value)*action_amplitude
   end

   return value
end


function sign(value)
   if value < 0 then
      return -1
   else
      return 1
   end
end