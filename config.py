
lr=0.00001 #local
#lr=0.00001

batch_size = 14 #local
#batch_size = 32
sequence_length = 36
w,h = 128, 128

num_queries = 4
skip_factor = 2

colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
colors_n = ["none"] + colors
shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]
shapes_n = ["None"] + shapes
belowAbove = ["standalone", "below", "above"]
actions = [-5,-2,-1,0,1,2,5]

#training_path = 'D:/training-data-relative-pos-no-obstacle/'
#training_path = 'D:/training-data-relative-pos-with-obstacle/'
#training_path = 'D:/onedrive-zhaw/OneDrive - ZHAW/training-data/new-training-data-with-obst-1/'
training_path = 'C:/Users/Dano/Documents/ZHAW/bachelor-thesis/training-data-plant/'
#training_path = 'C:/Users/Dano/Documents/ZHAW/bachelor-thesis/training-data-no-obstacle/'
#training_path = '/Users/ralph/Documents/Blender/training-data-relative-pos-no-obstacle/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes_obstacles/'
temp_path = './.temp/'
