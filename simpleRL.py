"""
Simple RL based on ground truths

"""
import os
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

SUB_DIR_IMAGE = 'recorded_cameras_head_camera_2_image_compressed'
FILENAME_FOR_REWARD = "recorded_button1_is_pressed.txt"
FILENAME_FOR_ACTION = "recorded_robot_limb_left_endpoint_action.txt"
FILENAME_FOR_STATE = "recorded_robot_limb_left_endpoint_state.txt"

path = "simpleData3D"
folders = filter(lambda x: os.path.isdir(os.path.join(path,x)), os.listdir(path))
# ------- pre-processing --------
# scaler = sklearn.preprocessing.StandardScaler()
# scaler.fit

# ------- create models  --------

class Estimator():
    # (action)-value function approximator
    def __init__(self):
        self.model = SGDRegressor(learning_rate = "constant")
        self.model.partial_fit(np.zeros((1,6)), [0])
    def predict(self, s, a):
        # feature = np.transpose(np.concatenate([s,a]))
        feature = np.concatenate([s,a])
        feature = np.transpose(feature[..., None])
        # print feature.shape
        score = self.model.predict(feature)
        return score
    def update(self, s, a, y):
        # feature = np.transpose(np.concatenate([s,a]))
        feature = np.concatenate([s,a])
        # feature = feature[..., None]
        feature = np.transpose(feature[..., None])
        self.model.partial_fit(feature, [y])
estimator = Estimator()
test_action = []
test_state = []
test_reward = []

for fld in reversed(folders):
    # ------- TODO  --------
    file_reward = open(os.path.join(path, fld, FILENAME_FOR_REWARD))
    file_action = open(os.path.join(path, fld, FILENAME_FOR_ACTION))
    file_state = open(os.path.join(path, fld, FILENAME_FOR_STATE))
    lines_reward = file_reward.readlines()[1:]
    lines_action = file_action.readlines()[1:]
    lines_state = file_state.readlines()[1:]
    reward = np.zeros(len(lines_reward))
    action = np.zeros((len(lines_action),3))
    state = np.zeros((len(lines_state),3))

    for i, l in enumerate(lines_reward):
        data = l.split(" ")
        reward[i] = data[1]
    for i, l in enumerate(lines_action):
        data = l.split(" ")
        action[i] = data[1:]
    for i, l in enumerate(lines_state):
        data = l.split(" ")
        state[i] = data[1:]

    # ------- TODO  --------
    gamma = 1.0
    for i in range(len(reward)-1):
        s = state[i]
        a = action[i]
        ns = state[i+1]
        na = action[i+1]
        target = reward[i] + gamma * estimator.predict(ns, na)
        estimator.update(s, a, target.ravel())
    test_action = action
    test_state = state
    test_reward = reward
    test_fld = fld
    # break

actions = test_action

actions =set(tuple(map(tuple,actions)))
z = []
for i, s in enumerate(test_state):
    z.append(max([estimator.predict(s, x)[0] for x in actions]))
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
print test_fld
for i, x in enumerate(test_reward):
    if x == 1:
        print i
plt.plot(z)
plt.show()
# ax = fig.add_subplot(111, projection='3d')
# Axes3D.plot_surface(test_state[:,0],test_state[:,1],test_state[:,2], z)
# plt.show()

# TODO: need refactoring to make it possible to read data into global variables
# TODO: find a way to plot 3D graphics that showing the value function as temperature
# TODO: 

"""
How can you make such a mistake?
Faster, faster!
What are you doing.
Faster, faster!
How can you make such a mistake?
How can you make such a mistake?
How can you make such a mistake?
What are you doing.
Hurry up!
Look, the time is ticking
Look, the time is ticking
"""
