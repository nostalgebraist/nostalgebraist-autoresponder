#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./decode.py /path/to/input.npz

import argparse
import numpy as np

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]

import encoder
from load_dataset import load_dataset

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('in_npz', metavar='IN.npz', type=str, help='Input file path')

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    chunks = load_dataset(enc, args.in_npz, args.combine)
    for chunk in chunks:
      text = enc.decode(chunk)
      print(text)
      sys.stdout.flush()

if __name__ == '__main__':
    main()
