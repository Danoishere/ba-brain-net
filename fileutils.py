import time
import os
import random
from multiprocessing import Process, Queue, Lock
from multiprocessing.pool import ThreadPool
import zipfile
import glob, os
import re
import json
import torch
import torch.nn as nn
import numpy as np
import config

training_path = config.training_path
temp_path = config.temp_path

batch_size = config.batch_size
sequence_length = config.sequence_length
w, h = config.w, config.h
zip_files = list(glob.glob(training_path + "*.zip"))

pool = ThreadPool(10)
process_suffix = str(os.getpid()) + "/" 

def extract_zip(file, to):
    start = time.time()
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(to + "/" + process_suffix)
    print("Unzip:", time.time() - start)


def vec_dist_loss(pos, pos_predict):
    vec_loss = torch.sqrt(torch.sum((pos - pos_predict)**2))
    return vec_loss


def col_loss(col, col_predict):
    col_loss = torch.sqrt(torch.sum((col - col_predict)**2))
    return col_loss


def reshape_frame(rgb, depth):
    frame_input = np.zeros((4,h,w))
    frame_input[0,:,:] = rgb[:,:,0]
    frame_input[1,:,:] = rgb[:,:,1]
    frame_input[2,:,:] = rgb[:,:,2]
    frame_input[3,:,:] = depth

    return frame_input


def load_batch():
    batch_x = np.zeros((sequence_length, batch_size, 4, h, w))
    batch_idx = 0

    scenes = []
    selected_files = random.sample(zip_files, batch_size)

    start = time.time()
    for file in selected_files:
        scene_id = os.path.basename(file).split(".")[0]
        extract_zip(file, temp_path)

        rgb_frames, depth_frames, scene_data = load_scene_data(scene_id)
        is_reversed = np.random.random() < 0.5

        frame_input = np.zeros((4,w,h))
        for frame in range(sequence_length):
            frame_idx = frame
            if is_reversed:
                frame_idx = -frame - 1

            rgb = rgb_frames[frame_idx]
            depth = depth_frames[frame_idx]
            frame_input = reshape_frame(rgb, depth)
            batch_x[frame, batch_idx, :,:,:] = frame_input

        if is_reversed:
            scene_data["cam_base_matricies"].reverse()
            scene_data["ss_objs"].reverse()

        scene_data["is_reversed"] = is_reversed
        scenes.append(scene_data)
        batch_idx += 1

    print("Elapsed:", time.time() - start)
    return batch_x, scenes

def load_scene_data(scene_id):
    json_file = scene_id + "-scene.json"
    np_file = scene_id + "-combined.npz"

    start = time.time()
    with np.load(temp_path + process_suffix +  np_file) as frames:
        rgb_frames = frames['rgb']
        depth_frames = frames['depth']

    with open(temp_path + process_suffix + json_file) as json_file_p:
        scene_data = json.load(json_file_p)


    print("Time for json and npz", time.time() - start)
    rgb_frames = rgb_frames/255.0
    depth_frames = np.tanh(0.2*depth_frames)
    depth_frames[np.isnan(depth_frames)] = 1.0

    os.remove(temp_path  + process_suffix + np_file)
    os.remove(temp_path  + process_suffix + json_file)

    return rgb_frames, depth_frames, scene_data
