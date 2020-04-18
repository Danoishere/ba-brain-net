import time
import os
import random
from multiprocessing import Process, Queue, Lock
from fileutils import load_batch
from train_autoencoder import train_autoencoder
from train_videornn import train_video_rnn
import argparse


 
# Producer function that places data on the Queue
def prod(queue, lock):
    # Synchronize access to the console
    with lock:
        print('Starting producer => {}'.format(os.getpid()))
         
    batch_nr = 0
    # Place our names on the Queue
    while True:
        with lock:
            print('Producing batch nr. {}'.format(batch_nr))
        batch = load_batch()
        queue.put(batch)

        batch_nr += 1
 
    # Synchronize access to the console
    with lock:
        print('Producer {} exiting...'.format(os.getpid()))
 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='BA>>Programm')
    parser.add_argument('--ae', help='Run auto-encoder', action='store_true')
    parser.add_argument('--vrnn', help='Video RNN', action='store_true')
    args = parser.parse_args()

    # Create the Queue object
    queue = Queue(maxsize=3)
     
    # Create a lock object to synchronize resource access
    lock = Lock()
    producer = Process(target=prod, args=(queue, lock))
    producer.start()

    if args.ae:
        train_autoencoder(queue, lock)
    if args.vrnn:
        train_video_rnn(queue, lock)

    producer.join()
 
    print('Parent process exiting...')