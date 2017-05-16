#!/usr/bin/python
#coding: utf-8

from __future__ import division
from torch.utils.serialization import load_lua

import numpy as np
import torch
from torchvision import transforms


import os

from PIL import Image

imagesFolder = 'simpleData3D'

modelString = 'dummyModelForDavid.t7'
model = load_lua(modelString)

reformatPipe = transforms.Compose([
    transforms.Scale(200),
    transforms.CenterCrop((200,200)),
    transforms.ToTensor()])

for seq in os.listdir(imagesFolder):
    strSeq = imagesFolder+'/'+seq+'/recorded_cameras_head_camera_2_image_compressed/'

    for imagePath in os.listdir(strSeq):
        if imagePath == 'time.txt':
            continue
            
        img = Image.open(strSeq+imagePath)
        img = reformatPipe(img)
        #NEED TO ADD NORMALIZATION AT A MOMENT

        representationOut = model.forward(img)
        print "representationOut",representationOut 
        
        
