
# lr=0.000005 #local
lr=0.00001

#batch_size = 16 #local
batch_size = 64
sequence_length = 36
w,h = 128, 128

num_queries = 8
skip_factor = 2

training_path = 'D:/training-data-relative-pos-no-obstacle/'
#training_path = '/Users/ralph/Documents/Blender/training-data-relative-pos-no-obstacle/'
#training_path = '/cluster/home/meierr18/BA_2020/training_scenes/'
temp_path = './.temp/'