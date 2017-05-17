require 'lfs'
require 'torch'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'

require 'const'

local imagesFolder = 'simpleData3D'
local modelString = 'Log/model2017_137__11_41_48.t7'

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
            local img=image.load(fullImagesPath,3,'float')

            img=image.scale(img,'200x200'):float():reshape(1,3,200,200):cuda()
            
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
