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

from tqdm import tqdm

training_path = config.training_path
temp_path = config.temp_path

batch_size = config.batch_size
sequence_length = config.sequence_length
w, h = config.w, config.h
zip_files = list(glob.glob(training_path + "*.zip"))

pool = ThreadPool(10)
process_suffix = str(os.getpid()) + "/" 

def extract_zip(file, to):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(to + "/unpacked")

for zip_file in tqdm(zip_files):
    extract_zip(zip_file, training_path)



