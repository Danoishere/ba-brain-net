
lr=0.000005 #local
#lr=0.00001

#batch_size = 16 #local
batch_size = 32
sequence_length = 36
w,h = 128, 128

num_queries = 8
skip_factor = 2

colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
colors_n = ["none", "red", "green", "blue", "yellow", "white", "grey", "purple"]
shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]
shapes_n = ["None", "Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]
belowAbove = ["standalone", "below", "above"]
actions = [-5,-2,-1,0,1,2,5]

#training_path = 'D:/training-data-relative-pos-no-obstacle/'
#training_path = 'D:/training-data-relative-pos-with-obstacle/'
#training_path = 'C:/Users/Dano/Documents/ZHAW/bachelor-thesis/training-data-no-obstacle/'
#training_path = '/Users/ralph/Documents/Blender/training-data-relative-pos-no-obstacle/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes/'
training_path = '/cluster/home/meierr18/BA_2020/training_scenes_obstacles/'
temp_path = './.temp/'
