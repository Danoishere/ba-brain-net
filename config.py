
lr=0.000005 #local
#lr=0.00001

batch_size = 2 #local
#batch_size = 64
sequence_length = 36
w,h = 128, 128

num_queries = 8
skip_factor = 2

colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]
belowAbove = ["standalone", "below", "above"]

#training_path = 'D:/training-data-relative-pos-no-obstacle/'
training_path = 'C:/Users/Dano/Documents/ZHAW/bachelor-thesis/training-data-no-obstacle/'
# training_path = '/Users/ralph/Documents/Blender/training-data-relative-pos-no-obstacle/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes/'
temp_path = './.temp/'

def set_rl_params():
    global lr, batch_size, sequence_length, w,h, num_queries,skip_factor, colors, shapes, belowAbove, training_path, temp_path
    batch_size = 1

