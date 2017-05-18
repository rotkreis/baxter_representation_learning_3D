--=============================================================
-- Constants file

-- All constants should be located here for now, in CAPS_LOCK
-- Ex : NB_EPOCHS = 10
-- Disclamer there still plenty of constants everywhere
-- Because it wasn't done like this before, so if you still
-- find global variable, that's normal. You can change it
-- and put it here.
--=============================================================
require 'lfs'
require 'cutorch'
torch.manualSeed(100)

DATA_FOLDER = 'simpleData3D'
PRELOAD_FOLDER = 'preload_folder/'
lfs.mkdir(PRELOAD_FOLDER)

LOG_FOLDER = 'Log/'
MODEL_PATH = LOG_FOLDER

MODEL_FILE_STRING  = MODEL_PATH..'13_09_adagrad4_coef1/Everything/Save13_09_adagrad4_coef1.t7'

MODEL_ARCHITECTURE_FILE = './models/topUniqueSimpler'

STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'


-- Create actions that weren't done by the robot
-- by sampling randomly states (begin point and end point)
-- Cannot be applied in every scenario !!!!
EXTRAPOLATE_ACTION = false
CLAMP_CAUSALITY = true

-- if you want to visualize images, use 'qlua' instead of 'th'
--===========================================================
VISUALIZE_IMAGES_TAKEN = false
VISUALIZE_CAUS_IMAGE = false
VISUALIZE_IMAGE_CROP = false
VISUALIZE_MEAN_STD = false

if VISUALIZE_CAUS_IMAGE or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD then
   --Creepy, but need a placeholder, to prevent many window to pop
   WINDOW = image.display(image.lena())
end
--===========================================================

RELOAD_MODEL = false

LR=0.0001
DIMENSION=3

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 2 -- TRYING TO HAVE BIGGER BATCH
NB_EPOCHS=20

IM_LENGTH = 200
IM_HEIGHT = 200

DATA_AUGMENTATION = 0.01
USE_CUDA = true

USE_SECOND_GPU = true

if USE_CUDA and USE_SECOND_GPU then
   cutorch.setDevice(2)
end

now = os.date("*t")

DAY = now.year..'_'..now.yday..'__'..now.hour..'_'..now.min..'_'..now.sec

NAME_SAVE= 'model'..DAY

if DATA_FOLDER == 'simpleData3D' then
   MIN_X = 0.42
   MAX_X = 0.8

   MIN_Y = -0.2
   MAX_Y = 0.7

   MIN_Z = -10
   MAX_Z = 10
end
