import argparse
use_cuda=False
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM ')
    parser.add_argument('--load_model_path',help='path to source trained model')
    parser.add_argument('--s_train_sent',help='path to source train dataset')
    parser.add_argument('--s_train_tree',help='path to source train dataset')
    parser.add_argument('--s_val_sent')
    parser.add_argument('--s_val_tree')
    parser.add_argument('--t_train_sent')
    parser.add_argument('--t_train_tree')
    parser.add_argument('--t_val_sent')
    parser.add_argument('--t_val_tree')
    parser.add_argument('--s_test_sent')
    parser.add_argument('--s_test_tree')
    parser.add_argument('--t_test_sent')
    parser.add_argument('--t_test_tree')
    parser.add_argument('--save_model_path',help='path to the saved model')
    parser.add_argument('--beta')
    args=parser.parse_args()
    return args

