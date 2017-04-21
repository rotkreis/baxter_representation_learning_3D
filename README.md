# baxter_representation_learning_3D

This project is about a paper written by Rico Jonschkowski and Oliver Brock. The goal is to learn a state representation based on images and robotics priors to make a network able to give high level information to another program which will make a robot learning tasks.

In this folder the network aims to learn a 3D representation of the hand position.

DATA:

Place your data (from GDrive folder) in the main folder. The data folder should be named "data_baxter".

MODEL:

The function "save_model" in script.lua saves models for each test. The tests done are defined in the list "Tests_Todo". Each test trains the model with a particular combination of priors but the best one used now is the one with all the priors. Once you have saved a trained model, load_model.lua loads it using the variable "name"
 


Note: This repo is an extension of https://github.com/Mathieu-Seurin/baxter_representation_learning_1D to the 3D case: unsupervised learning of states for 3D representation learning


