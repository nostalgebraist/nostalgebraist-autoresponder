import argparse
import json
import os
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2

import model, sample, encoder
from load_dataset import load_dataset, Sampler
from accumulate import AccumulatingOptimizer
import memory_saving_gradients

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'


parser = argparse.ArgumentParser(
    description='Review random chunks from a dataset (helpful for reviewing the distribution of different text genres, etc).',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='774M', help='Pretrained model name (used to get BPE encoder only)')

def main():
    args = parser.parse_args()

    enc = encoder.get_encoder(args.model_name)

    print('Loading dataset...')
    chunks = load_dataset(enc, args.dataset, args.combine)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')
    #ix = 0
    try:
        while True:
            encoded_sample = data_sampler.sample(1024)
            #encoded_sample = data_sampler.chunks[ix]
            text = enc.decode(encoded_sample)
            print("\n\n===SAMPLE===")
            print(text)
            _ = input("Press any key >>> ")
            #ix += 1
    except KeyboardInterrupt:
        print('interrupted')

if __name__ == '__main__':
    main()
