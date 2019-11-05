import argparse
import os

from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--batch-size', help='batch size')
    parser.add_argument('--data-path', help='dataset path')
    
    return parser.parse_args()

def train(model, epoch, train_data, optimizer, logger):
    
