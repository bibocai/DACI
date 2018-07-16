# coding: utf-8



import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM')
    parser.add_argument('--domain',help='path to source trained model')
    # parser.add_argument('--model',help='prop_relation_or_no_relation')
    parser.add_argument('--output_dir',help='output_dir')
    args=parser.parse_args()
    return args