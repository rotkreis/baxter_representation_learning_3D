require 'const'
require 'Get_Images_Set'

--============== Tools to get action from get state ===========
--=============================================================

-- Making actions not be the same but close enough for the continous handling of priors
function actions_are_close_enough(action1,action2)
  local close_enough = true
  --for each dim, check that the magnitude of the action is close
  for dim=1,DIMENSION_IN do
     close_enough = close_enough and arrondit(action1[dim] - action2[dim]) < CLOSE_ENOUGH_PRECISION_THRESHOLD
  end
  return close_enough
end

function get_one_random_Prop_Set_and_actions(Infos1)
   return get_two_Prop_Pair_and_actions(Infos1,Infos1)
end

---------------------------------------------------------------------------------------
-- Function : get_two_Prop_Pair_and_action_deltas(txt1, txt2,use_simulate_images)
-- Input (txt1) : path of the file of the first list of joint
-- Input (txt2) : path of the file of the second list of joint
-- Input (use_simulate_images) : boolean variable which say if we use or not simulate images (we need this information because the data is not formated exactly the same in the txt file depending on the origin of images)
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Traininng with prop prior.
-- Returns a Lua table with 4 images and the 2 actions derived from the states
-- {
--   im3 : 19
--   im2 : 46
--   act2 :
--     {
--       1 : 0.051146118604
--       2 : -0.04237182035
--       3 : 0.0452409758369
--     }
--   act1 :
--     {
--       1 : 0.045235814614
--       2 : -0.051011220414
--       3 : 0.0505784082665
--     }
--   im4 : 20
--   im1 : 45
-- }
-- The variation of joint for one pair should be close enough (<CLOSE_ENOUGH_PRECISION_THRESHOLD) in continuous actions, to the variation for the second
---------------------------------------------------------------------------------------
function get_two_Prop_Pair_and_actions(Infos1, Infos2)
  --  print('Infos1')
  --  print(Infos1)
   local watchDog=0
   local size1=#Infos1[1]
   local size2=#Infos2[1]

   local vector=torch.randperm(size2-1)

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
               print('get_two_Prop_Pair_and_actions')
               print(action1)
               print(action2)
               if actions_are_close_enough(action1, action2) then --is_same_action(action1, action2) then
                  return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end, act1=action1, act2=action2}
               else
                  print ('get_two_Prop_Pair_and_actions did not find actions close enough, rise the CLOSE_ENOUGH_PRECISION_THRESHOLD '..CLOSE_ENOUGH_PRECISION_THRESHOLD..' for DIMENSION_IN: '..DIMENSION_IN)
               end
            end
         else --USE THE NEXT IMAGE IN THE SEQUENCE
            id_second_action_end=id_second_action_begin+1
            action2 = action_amplitude(Infos2, id_second_action_begin, id_second_action_end)
            print('get_two_Prop_Pair_and_actions')
            print(action1)
            print(action2)
            if actions_are_close_enough(action1, action2) then
               return {im1=id_ref_action_begin,im2=id_ref_action_end,im3=id_second_action_begin,im4=id_second_action_end, act1=action1, act2=action2}
            else
               print ('get_two_Prop_Pair_and_actions did not find actions close enough, rise the CLOSE_ENOUGH_PRECISION_THRESHOLD '..CLOSE_ENOUGH_PRECISION_THRESHOLD..' for DIMENSION_IN: '..DIMENSION_IN)
            end
         end
      end
      watchDog=watchDog+1
   end
   error("PROP WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end

---------------------------------------------------------------------------------------
-- Function : get_one_random_Caus_Set_and_actions(Infos1, Infos2)
-- Input  -- I need to search images representing a starting state.
-- then the same action applied to this state (the same variation of joint) should lead to a different reward.
-- for instance, we choose as reward the fact of having a joint value = 0
-- NB : the two states will be took from different list but the two list can be the same
-- Output : structure with 4 indices which represente a quadruplet (2 Pair of images from 2 different list) for Training with prop prior,
--  and an array of the delta in between actions (the distance in between 2 actions as Euclidean distance)
-- The variation of joint for one pair should be close enough (<CLOSE_ENOUGH_PRECISION_THRESHOLD) in continuous actions, to the variation for the second
---------------------------------------------------------------------------------------
function get_one_random_Caus_Set_and_actions(Infos1, Infos2)
   local watchDog=0

   local size1=#Infos1[1]
   local size2=#Infos2[1]
   vector=torch.randperm(size2-1)

   while watchDog<100 do
      repeat
         id_ref_action_begin= torch.random(1,size2-1)

         if EXTRAPOLATE_ACTION then --Look at const.lua for more details about extrapolate
            id_ref_action_end  = torch.random(1,size2)
         else
            id_ref_action_end  = id_ref_action_begin+1
         end

      until(Infos2.reward[id_ref_action_begin]==0 and Infos2.reward[id_ref_action_end]~=0)

      reward1 = Infos2.reward[id_ref_action_end]

      if VISUALIZE_CAUS_IMAGE then
         visualize_image_from_seq_id(indice2,id_ref_action_begin,id_ref_action_end)
      end

      action1 = action_amplitude(Infos2, id_ref_action_begin, id_ref_action_end)

      if CLAMP_CAUSALITY and not EXTRAPOLATE_ACTION then
         -- WARNING, THIS IS DIRTY, need to do continous prior
         for dim=1,DIMENSION_IN do
            action1[dim]=clamp_causality_prior_value(action1[dim])
         end
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

         if Infos1.reward[id_second_action_begin]==0 and Infos1.reward[id_second_action_end]~=reward1 then
            action2 = action_amplitude(Infos1, id_second_action_begin, id_second_action_end)
            --print("action2",action2.x,action2.y,action2.z)

            if actions_are_close_enough(action1, action2) then
               return {im1=id_second_action_begin,im2=id_ref_action_begin, im3=id_second_action_end, im4=id_ref_action_end, act1=action1, act2=action2}
            end
         end
      end
      watchDog=watchDog+1
   end
   error("CAUS WATCHDOG ATTACK!!!!!!!!!!!!!!!!!!")
end
