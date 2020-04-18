
import zipfile
import glob, os
import re
import json
import torch
import torch.nn as nn
import numpy as np
from autoencoder import ConvAutoencoder
import matplotlib.pyplot as plt
from random import shuffle
from time import time


def train_autoencoder(queue, lock, load_model=True):
    lr=0.001
    batch_size = 64
    sequence_length = 36
    w, h = 128, 128
    colors = ["red", "green", "blue", "yellow", "white", "grey", "purple"]

    cae = ConvAutoencoder().cuda()
    if load_model:
        cae.load_state_dict(torch.load('cae-model'))
        cae.eval()
        # torch.save(cae, "cae-model-full.mdl")

    params = []
    params += list(cae.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)
    criterion = nn.MSELoss()

    losses = []
    episodes = []

    episode = 0

    plt.ion()
    plt.imshow(np.zeros((128, 128, 3)))
    plt.show()

    for repetition in range(100):
       
        batch_nr = 0
        while True:
            load_start = time()
            batch_x, _ = queue.get()
            print("Load time for", batch_size, "images:", time() - load_start)

            p_start = time()
            # Pass batch frame by frame
            for frame in range(sequence_length):
                optimizer.zero_grad()
                frame_input = torch.tensor(batch_x[frame], requires_grad=True).float().cuda()
                frame_output = torch.tensor(batch_x[frame]).float().cuda()
                output = cae(frame_input)
                loss = criterion(output, frame_output)
                loss.backward()

                if frame == 0:
                    img = frame_input.cpu().detach().numpy()[0].swapaxes(0, 2)
                    img = np.rot90(img[:,:,:4], 3)
                    img = np.fliplr(img)
                    img *= 255.0
                    img = img.astype(np.uint8)

                    imgo = output.cpu().detach().numpy()[0].swapaxes(0, 2)
                    imgo = np.rot90(imgo[:,:,:4], 3)
                    imgo = np.fliplr(imgo)
                    imgo *= 255.0
                    imgo = imgo.astype(np.uint8)

                    plt.subplot(3,1,1)
                    plt.imshow(img[:,:,3])
                    plt.subplot(3,1,2)
                    plt.imshow(imgo[:,:,3])
                    plt.subplot(3,1,3)
                    plt.plot(episodes, losses)

                    plt.show(block=False)
                    plt.pause(0.001)

                optimizer.step()

            print("Processing time for", batch_size, "images:", time() - p_start)
            print('Repetition', repetition, ', Batch', batch_nr, ', Episode', episode, ', Loss Col.:', loss.item())

            episodes.append(episode)
            losses.append(loss.item())
            episode += 1
            batch_nr += 1

            if episode % 100 == 0:
                torch.save(cae, "cae-model-full-" + episode + ".mdl")
                # torch.save(cae.state_dict(), 'cae-model')


    plt.plot(episodes, losses)
    plt.show()
        
