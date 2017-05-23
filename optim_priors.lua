function doStuff_temp(Models,criterion,Batch,coef)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, Model, Model2, State1, State2

   local batchSize = Batch:size(2)

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
   end

   Model=Models.Model1
   Model2=Models.Model2
   State1=Model:forward(im1)
   State2=Model2:forward(im2)

   assert(batchSize==State1:size(1), "Batch Size changed during 'forward method, maybe a nn.view is done badly ...")

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   loss=criterion:forward({State2,State1})
   GradOutputs=criterion:backward({State2,State1})

   -- calculer les gradients pour les deux images
   Model:backward(im1,coef*GradOutputs[2])
   Model2:backward(im2,coef*GradOutputs[1])
   return loss, coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Caus(Models,criterion,Batch,coef)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, Model, Model2, State1, State2

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
   end

   Model=Models.Model1
   Model2=Models.Model2

   State1=Model:forward(im1)
   State2=Model2:forward(im2)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2})
   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2}, torch.ones(1))

   -- compute the gradients for the two images
   Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
   Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Prop(Models,criterion,Batch, coef)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State1=Model:forward(im1)
   State2=Model2:forward(im2)
   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})

   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4},torch.ones(1))

   Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
   Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
   Model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
   Model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))

   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Rep(Models,criterion,Batch, coef)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})

   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4}, torch.ones(1))

   Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
   Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
   Model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
   Model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))

   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

----CONTINUOUS ACTION PRIORS VERSION

function actions_distance(action1, action2)
  -- Returns a double indicating absolute value sum of the distance in each dimension among actions
  local distance = 0
  --for each dim, check that the magnitude of the action is close
  for dim=1, DIMENSION_IN do
     distance = distance + arrondit(action1[dim] - action2[dim])
  end
  return distance
end

function get_continuous_factor_term(action1, action2)
  -- print('get_continuous_factor_term with sigma='.. GAUSSIAN_SIGMA..' and action1 :')
  -- print(action1)
  -- print(DIMENSION_IN)
  --squared_distance = nn.Square()(actions_distance) --discounted_squared_distance = nn.CMulTable()({squared_distance, 1/GAUSSIAN_SIGMA})
  --  continuous_loss = nn.CMulTable()({continuous_factor_term, sqr})
  --  out = nn.Sum(1,1)(continuous_loss)
  -- print('continuous_factor_term: ')
  -- print(continuous_factor_term)
  return math.exp((-1 * actions_distance(action1, action2))/GAUSSIAN_SIGMA)
end

function doStuff_Caus_continuous(Models,criterion,Batch,coef, action1, action2)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, Model, Model2, State1, State2

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
   end

   Model=Models.Model1
   Model2=Models.Model2

   State1=Model:forward(im1)
   State2=Model2:forward(im2)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2})
   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2}, torch.ones(1))

   continuous_factor_term = get_continuous_factor_term(action1, action2)
   -- compute the gradients for the two images
   Model:backward(im1, continuous_factor_term * coef*GradOutputs[1]/Batch[1]:size(1))
   Model2:backward(im2, continuous_factor_term * coef*GradOutputs[2]/Batch[1]:size(1))
   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Prop_continuous(Models,criterion,Batch, coef, action1, action2)
  -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State1=Model:forward(im1)
   State2=Model2:forward(im2)
   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})
   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4},torch.ones(1))

   continuous_factor_term = get_continuous_factor_term(action1, action2)
   --print('im1') --[torch.DoubleTensor of size 2x3x200x200]   print(im1)
   --print('coef')   print(coef) --1
   --print('continuous_factor_term')
   --print(continuous_factor_term) --nngraph.Node
   --print('GradOutputs[2]/Batch[1]:size(1)')
  --  0.0407 -0.0330 -0.0476
  --   0.2309 -0.1391 -0.5333
  --  [torch.DoubleTensor of size 2x3]
  --outTot=nn.Sum(1,1)(nn.CMulTable()({out1, out2}))
   --common_factor_times_coef = nn.CMul()({continuous_factor_term, coef})

   --common_factor = nn.CMulTable()({common_factor_times_coef, GradOutputs[1]/Batch[1]:size(1)})
   Model:backward(im1, continuous_factor_term * coef *GradOutputs[1]/Batch[1]:size(1))--Model:backward(im1,continuous_factor_term * coef*GradOutputs[1]/Batch[1]:size(1))
   --common_factor = nn.CMulTable()({common_factor_times_coef, GradOutputs[2]/Batch[1]:size(1)})
   Model2:backward(im2, continuous_factor_term * coef *GradOutputs[2]/Batch[1]:size(1))
   --common_factor = nn.CMulTable()({common_factor_times_coef, GradOutputs[3]/Batch[1]:size(1)})
   Model3:backward(im3,continuous_factor_term * coef *GradOutputs[3]/Batch[1]:size(1))
   --common_factor = nn.CMulTable()({common_factor_times_coef, GradOutputs[4]/Batch[1]:size(1)})
   Model4:backward(im4,continuous_factor_term * coef *GradOutputs[4]/Batch[1]:size(1)) --Model4:backward(im4,continuous_factor_term * coef*GradOutputs[4]/Batch[1]:size(1))

   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Rep_continuous(Models,criterion,Batch, coef, action1, action2)
  -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State1=Model:forward(im1)
   State2=Model2:forward(im2)
   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})

   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4}, torch.ones(1))

   continuous_factor_term = get_continuous_factor_term(action1, action2)
   Model:backward(im1,continuous_factor_term * coef*GradOutputs[1]/Batch[1]:size(1))
   Model2:backward(im2,continuous_factor_term * coef*GradOutputs[2]/Batch[1]:size(1))
   Model3:backward(im3,continuous_factor_term * coef*GradOutputs[3]/Batch[1]:size(1))
   Model4:backward(im4,continuous_factor_term * coef*GradOutputs[4]/Batch[1]:size(1))

   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end
