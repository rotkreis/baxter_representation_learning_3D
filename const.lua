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

--=====================================
--DATA AND LOG FOLDER NAME etc..
--====================================
--DATA_FOLDER = 'simpleData3D'
DATA_FOLDER = 'mobileRobot'

print("============ DATA USED =========\n",
      DATA_FOLDER,
      "\n================================")

PRELOAD_FOLDER = 'preload_folder/'
lfs.mkdir(PRELOAD_FOLDER)

LOG_FOLDER = 'Log/'
MODEL_PATH = LOG_FOLDER

MODEL_ARCHITECTURE_FILE = './models/topUniqueSimplerWOTanh'
--MODEL_ARCHITECTURE_FILE = './models/topUniqueSimpler'

STRING_MEAN_AND_STD_FILE = PRELOAD_FOLDER..'meanStdImages_'..DATA_FOLDER..'.t7'

now = os.date("*t")
DAY = now.year..'_'..now.yday..'__'..now.hour..'_'..now.min..'_'..now.sec
NAME_SAVE= 'model'..DAY


--===========================================================
-- if you want to visualize images, use 'qlua' instead of 'th'
--===========================================================
VISUALIZE_IMAGES_TAKEN = false
VISUALIZE_CAUS_IMAGE = false
VISUALIZE_IMAGE_CROP = false
VISUALIZE_MEAN_STD = false

if VISUALIZE_IMAGES_TAKEN or VISUALIZE_CAUS_IMAGE or VISUALIZE_IMAGE_CROP or VISUALIZE_MEAN_STD then
   --Creepy, but need a placeholder, to prevent many window to pop
   WINDOW = image.display(image.lena())
end


--==================================================
-- Hyperparams : Learning rate, batchsize, USE_CUDA etc...
--==================================================
RELOAD_MODEL = false

EXTRAPOLATE_ACTION = false
LR=0.001

SGD_METHOD = 'adam' -- Can be adam or adagrad
BATCH_SIZE = 3 -- TRYING TO HAVE BIGGER BATCH
NB_EPOCHS=2

IM_LENGTH = 200
IM_HEIGHT = 200

DATA_AUGMENTATION = 0.01
NORMALIZE_IMAGE = true

COEF_TEMP=0.3
COEF_PROP=0.3
COEF_REP=0.3
COEF_CAUS=1

USE_CUDA = false--true

USE_SECOND_GPU = true

if USE_CUDA and USE_SECOND_GPU then
   cutorch.setDevice(2)
end

--======================================================
--Continuous actions SETTINGS
--======================================================

USE_CONTINUOUS = false --Todo, a switch between those two ?  -- Requires calling getRandomBatchFromSeparateListContinuous instead of getRandomBatchFromSeparateList
ACTION_AMPLITUDE = 0.01
-- The following parameter eliminates the need of finding close enough actions for assessing all priors except for the temporal.one.
-- If the actions are too far away, they will make the gradient 0 and will not be considered for the update rule
GAUSSIAN_SIGMA = 0.1

--================================================
-- dataFolder specific constants : filename, dim_in, dim_out
--===============================================
if DATA_FOLDER == 'simpleData3D' then
   -- Create actions that weren't done by the robot
   -- by sampling randomly states (begin point and end point)
   -- Cannot be applied in every scenario !!!!

   DEFAULT_PRECISION = 0.05
   CLAMP_CAUSALITY = true

   MIN_TABLE = {0.42,-0.2,-10} -- for x,y,z
   MAX_TABLE = {0.8,0.7,10} -- for x,y,z

   DIMENSION_IN = 3
   DIMENSION_OUT= 3

   REWARD_INDICE = 2

   INDICE_TABLE = {2,3,4} --column indice for coordinate in state file (respectively x,y,z)

   DEFAULT_PRECISION = 0.05 -- for 'arrondit' function
   FILENAME_FOR_REWARD = "is_pressed"
   FILENAME_FOR_ACTION = "endpoint_action"
   FILENAME_FOR_STATE = "endpoint_state"

   SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'

elseif DATA_FOLDER == 'mobileRobot' then
   DEFAULT_PRECISION = 0.1
   CLAMP_CAUSALITY = false

   MIN_TABLE = {-10000,-10000} -- for x,y
   MAX_TABLE = {10000,10000} -- for x,y

   DIMENSION_IN = 2
   DIMENSION_OUT= 4

   REWARD_INDICE = 1
   INDICE_TABLE = {1,2} --column indice for coordinate in state file (respectively x,y)

   FILENAME_FOR_ACTION = "action"
   FILENAME_FOR_STATE = "state"
   FILENAME_FOR_REWARD = "reward"

   SUB_DIR_IMAGE = 'recorded_camera_top'

elseif DATA_FOLDER == 'realBaxterPushingObjects' then
  -- Leni's real Baxter data on  ISIR dataserver. It is named "data_archive_sim_1".
  DEFAULT_PRECISION = 0.1
  -- CLAMP_CAUSALITY = false
  --
  -- MIN_TABLE = {-10000,-10000} -- for x,y
  -- MAX_TABLE = {10000,10000} -- for x,y
  --
  -- DIMENSION_IN = 2
  -- DIMENSION_OUT= 4
  --
  -- REWARD_INDICE = 1
  -- INDICE_TABLE = {1,2} --column indice for coordinate in state file (respectively x,y)
  --
  -- FILENAME_FOR_ACTION = "action"
  -- FILENAME_FOR_STATE = "state"
  -- FILENAME_FOR_REWARD = "reward"
  --
  -- SUB_DIR_IMAGE = 'recorded_camera_top'

else
  print("No supported data folder provided, please add either of simpleData3D, mobileRobot or Leni's realBaxterPushingObjects")
  os.exit()
end


print("\n USE_CUDA ",USE_CUDA," \n USE_CONTINUOUS ACTIONS: ",USE_CONTINUOUS)
