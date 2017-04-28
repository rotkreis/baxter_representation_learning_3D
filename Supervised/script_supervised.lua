require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'math'
require 'string'
require 'cunn'
require 'nngraph'

require '../MSDC'
require '../functions.lua'
require '../printing.lua'
require "../Get_Images_Set"
require '../priors'

--Different models in the way the FM (feature map) is constructed:
--Using one feature map for each dimension (x,y,z) learned ("topTripleFM_Split.lua")
--Using a shared top feature map for the three dimensions ("topUniqueFM_Deeper2.lua")
--The model minimalNetModel.lua serves to test a small network to run on cpu only tests

local Path="../data_baxter"
--local Path="../data_baxter_short_seqs" -- Shorter sequences dataset
useCUDA = false
local UseSecondGPU= false --true
if not useCUDA then
	UseSecondGPU = false
	nb_part=100
	model_file='../models/minimalNetModel'
else
	nb_part = 50
	model_file= '../models/topTripleFM_Split'
end
-- if UseSecondGPU then
-- 	cutorch.setDevice(2)
-- end
print('Running main script with useCUDA flag: '..tostring(useCUDA))
print('Running main script with useSecondGPU flag: '..tostring(UseSecondGPU))
--made global for logging:
LR=0.005
print('nb_parts per batch: '..nb_part.." LearningRate: "..LR.." Using data folder: "..Path)


function Training(Models,Mode,batch,label,criterion,coef,LR)
	local LR=LR or 0.001
	local mom=0.0--9
        local coefL2=0,0
	-- just in case:
	collectgarbage()

	if useCUDA then
    batch=batch:cuda()
  end
	Label=torch.Tensor(2,3) --pb of batch sorry for bad programming issue
	Label[1]=label
	Label[2]=label

	if useCUDA then
		Label=Label:cuda()
	end
	 -- reset gradients
	State1=Model:forward(batch)
	if useCUDA then
			criterion=criterion:cuda()
	end

	output=criterion:forward({State1, Label})
	Model:zeroGradParameters()
	GradOutputs=criterion:backward({State1, Label})
	Model:backward(batch,GradOutputs[1])

	Model:updateParameters(LR)
	--optimState={learningRate=LR}
	--parameters, loss=optim.adagrad(feval, parameters, optimState)

	return output
end


function show_figure_sup(list_out1, Name, Variable_Name)
	local Variable_Name=Variable_Name or 'out1'
	-- log results to files
	local accLogger = optim.Logger(Name)


	for i=1, #list_out1 do
	-- update logger
		accLogger:add{[Variable_Name.."-DIM-1"] = list_out1[i][1],
				[Variable_Name.."-DIM-2"] = list_out1[i][2],
				[Variable_Name.."-DIM-3"] = list_out1[i][3]}
	end
	-- plot logger
	accLogger:style{[Variable_Name.."-DIM-1"] = '-',
			[Variable_Name.."-DIM-2"] = '-',
			[Variable_Name.."-DIM-3"] = '-'}

	accLogger.showPlot = false
	accLogger:plot()
end

function load_Part_supervised(list,txt_state,im_lenght,im_height,nb_part,part, coef_DA)
	local x=2
	local y=3
	local z=4
	local Data={Images={},Labels={}}
	local list_lenght = torch.floor(#list/nb_part)
	local start=list_lenght*part +1
	local tensor, label=tensorFromTxt(txt_state)
	for i=start, start+list_lenght do
		local Label=torch.Tensor(3)
		table.insert(Data.Images,getImage(list[i],im_lenght,im_height,coef_DA))
		Label[1]=tensor[i][x]
		Label[2]=tensor[i][y]
		Label[3]=tensor[i][z]
		table.insert(Data.Labels,Label*10)
	end
	return Data
end

function Print_Supervised(Model,Data, name, Log_Folder,criterion)
	local list_out1={}
	local sum_loss=0
	local loss=0

	if useCUDA then
			criterion=criterion:cuda()
	end

	for i=1, #Data.Images do --#imgs do
		image1=Data.Images[i]
		if useCUDA then
				Label=Data.Labels[i]:cuda()
		else
				Label=Data.Labels[i]
		end

		Data1=torch.Tensor(2,3,200,200)
		Data1[1]=image1
		Data1[2]=image1
		if useCUDA then
				Model:forward(Data1:cuda())
		else
				Model:forward(Data1)
		end
		loss=loss+criterion:forward({Model.output[1], Label})
		local State1= torch.Tensor(3)
		State1:copy(Model.output[1])
		table.insert(list_out1,State1)
	end
	Correlation, mutual_info=print_correlation(Data.Labels,list_out1,3)
	show_figure_sup(list_out1, Log_Folder..'state'..name..'.log')
	return loss/#Data.Images, mutual_info, Correlation
end

function train_Epoch(Models,Log_Folder,LR)
	local BatchSize=1
	local nbEpoch=1000
	local list_MI= {}
	local name='Save'..day
	local name_save=Log_Folder..name..'.t7'
	local criterion=nn.MSDCriterion()
	local list_errors={}
	print ("folders to process: ")
	print (nbList)

	indice_test= nbList --4
	part_test=1
	print('list_folders_images size: '..#list_folders_images)
	--print('list_folders_images[indice_test]: '..list_folders_images[indice_test])

	local list_test=images_Paths(list_folders_images[indice_test])
	local txt_test=list_txt_state[indice_test]
	Data_test=load_Part_supervised(list_test,txt_test,image_width,image_height,nb_part,part_test,0)
	local sum_loss=0
	local Loss_Train={}
	local Loss_Valid={}
	show_figure_sup(Data_test.Labels, Log_Folder..'The_Truth.Log')

	Print_Supervised(Models, Data_test,"First_Test",Log_Folder,criterion)

	for epoch=1, nbEpoch do
		sum_loss=0
		print('--------------Epoch : '..epoch..' ---------------')
		print(nbList..' : sequences')
		indice1=torch.random(1,nbList-1)
		indice1=4
		txt_state=list_txt_state[indice1]

		--local nb_part=50
		local part=torch.random(2,nb_part-1)-- part 0 contain void images, 1 is for test
		local list=images_Paths(list_folders_images[indice1])

		Data=load_Part_supervised(list,txt_state,image_width,image_height,nb_part,part,0.1)
		local nb_passage=10
		for j=1, nb_passage do
			for i=1, #Data.Images do
				batch=torch.Tensor(2,3, 200, 200)
				batch[1]=Data.Images[i]
				batch[2]=Data.Images[i]
				loss=Training(Models,Mode,batch,Data.Labels[i],criterion,coef,LR)
				sum_loss=sum_loss+loss
			end
				xlua.progress(j, nb_passage)
		end

		save_model(Model,name_save)
		loss_test, mutual_info, corr=Print_Supervised(Model,Data_test,name..epoch.."_Test",Log_Folder,criterion)

		table.insert(Loss_Train,sum_loss/(#Data.Images*nb_passage))
		table.insert(Loss_Valid,loss_test)
		show_loss(Loss_Train,Loss_Valid, Log_Folder..'Mean_loss.log')

		table.insert(list_MI,mutual_info)
		show_MI(list_MI, Log_Folder..'Mutuelle_Info.log')
		Print_Corr(corr,epoch,Log_Folder)
	end

end


day="21-10"
local Dimension=3
local Log_Folder='./Log/'..day..'/'
name_load='./Log/Save/'..day..'.t7'
list_folders_images, list_txt_action,list_txt_button, list_txt_state=Get_HeadCamera_View_Files(Path)
image_width=200
image_height=200

nbList= #list_folders_images
print('list_folders_images=')
print(list_folders_images)
torch.manualSeed(123)

--NOTE: model_file needs to be a global variable, otherwise we get:
-- /home/natalia/torch/install/bin/luajit: /home/natalia/torch/install/share/lua/5.1/trepl/init.lua:347: attempt to index local 'name' (a nil value)
-- stack traceback:
-- 	/home/natalia/torch/install/share/lua/5.1/trepl/init.lua:347: in function 'require'
-- 	script_supervised.lua:236: in main chunk
-- 	[C]: in function 'dofile'
-- 	...alia/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th:150: in main chunk
-- 	[C]: at 0x00406670

require(model_file)
Model=getModel(Dimension)
if useCUDA then
     Model:cuda()
end
parameters,gradParameters = Model:getParameters()
print("Training epoch...Current Log_Folder : "..Log_Folder)
train_Epoch(Model,Log_Folder,LR)

imgs={} --memory is free!!!!!
