
lr=0.000005 #local
#lr=0.00001

batch_size = 1 #local
#batch_size = 64
sequence_length = 36
w,h = 128, 128

num_queries = 8
skip_factor = 2
init_frames = 4

colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]
shapes = ["Cube", "CubeHollow", "Diamond", "Cone", "Cylinder"]
belowAbove = ["standalone", "below", "above"]
actions = ["5 left", "2 left", "1 left", "stay here","1 right","2 right","5 right"]

#training_path = 'D:/training-data-relative-pos-no-obstacle/'
training_path = 'C:/Users/Dano/Documents/ZHAW/bachelor-thesis/training-data-no-obstacle/'
#training_path = '/Users/ralph/Documents/Blender/training-data-relative-pos-no-obstacle/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes/'
temp_path = './.temp/'

BUFFER_SIZE = int(1e5)  #replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

eps_start=1.0
eps_end = 0.01
eps_decay=0.996