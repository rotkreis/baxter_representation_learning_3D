require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'

require 'const'
require 'functions'

local imagesFolder = 'simpleData3D'
local modelString = 'Log/model2017_137__18_20_38.t7'

local model = torch.load(modelString):cuda()
outStr = ''

for seqStr in lfs.dir(imagesFolder) do
   if string.find(seqStr,'record') then
      print("Sequence : ",seqStr)
      local imagesPath = imagesFolder..'/'..seqStr..'/recorded_cameras_head_camera_2_image_compressed'
      for imageStr in lfs.dir(imagesPath) do
         if string.find(imageStr,'jpg') then
            local fullImagesPath = imagesPath..'/'..imageStr
            local reprStr = ''
            img = getImageFormated(fullImagesPath):cuda():reshape(1,3,200,200)
            repr = model:forward(img)
            for i=1,repr:size(2) do
               reprStr = reprStr..repr[{1,i}]..' '
            end
            outStr = outStr..fullImagesPath..' '..reprStr..'\n'
            
         end
      end
   end
end        
        
file = io.open('saveImagesAndRepr.txt', 'w')
file:write(outStr)
