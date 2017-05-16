import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import shutil


# Some parameters
nbr_neighbors=2
data_file="images_and_repr_file.txt"


#reading data
file  = open(data_file, "r")

images=[]
states=[]
for line in file:
	words = line.split()
	images.append(words[0])
	states.append(words[1:])

dim_state= len(states[0])

#Compute nearest neighbors
nbrs = NearestNeighbors(n_neighbors=(nbr_neighbors+1), algorithm='ball_tree').fit(states)
distances, indices = nbrs.kneighbors(states)

#Generate mosaics
shutil.rmtree('NearestNeighbors', 1)
os.mkdir('NearestNeighbors')

for img_name,id,dist,state in zip(images,indices,distances,states):
	base_name= os.path.splitext(os.path.basename(img_name))[0]
	print('Processing ' + base_name + '...')
	fig = plt.figure()
	fig.set_size_inches(6*(nbr_neighbors+1), 6)
	a=fig.add_subplot(1,nbr_neighbors+1,1)
	a.axis('off')
	img = mpimg.imread(img_name)
	imgplot = plt.imshow(img)
	state_str='State: [' + ",".join(['{:.3f}'.format(float(x)) for x in state]) + "]"
	a.set_title(state_str)
	for i in range(0,nbr_neighbors):
		a=fig.add_subplot(1,nbr_neighbors+1,i+2)
		img = mpimg.imread(images[id[i+1]])
		imgplot = plt.imshow(img)
		dist_str = 'Dist: ' + '{:.5f}'.format(dist[i+1])
		a.set_title(dist_str)
		a.axis('off')
		
	plt.tight_layout()
	plt.savefig('NearestNeighbors/' + base_name + 'Neigbors.png',bbox_inches='tight')
