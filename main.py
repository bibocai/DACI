#-*- coding:utf-8 -*-

from train_model import Trainer,genDataset
import time
from test_pytorch_lstm import TreeLstm
import torch
from config import parse_args

input_size = 100
hidden_size = 200
batch_size = 128

# 把全局变量放到config里面去

if __name__ == '__main__':

    args = parse_args()
    model = TreeLstm(input_size,hidden_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9,weight_decay=1e-8)
    trainer = Trainer(model,criterion,optimizer,batch_size)
    train_dataset = genDataset(args.s_train_sent,args.s_train_tree)
    val_dataset = genDataset(args.s_val_sent,args.s_val_tree)
    test_dataset = genDataset(args.s_test_sent,args.s_test_tree)
    since = time.time()
    best_acc = 0.0
    best_model_wts = model.state_dict()

    for epoch in range(10):
        train_loss          = trainer.train(train_dataset)
        val_loss ,val_accu  = trainer.test(val_dataset)
        print('ValLoss: {:.4f} ValAcc: {:.4f}'.format(
             val_loss, val_accu))

        # val_loss, val_accu = trainer.test(t_val_dataset)

        print('--------------train------------------')
        print('TrainLoss: {:.4f}'.format(
              train_loss))
        print('--------------val--------------------')
        print('TrainLoss: {:.4f} ValAcc: {:.4f}'.format(
              val_loss, val_accu)) 
        if val_accu > best_acc:
            best_acc = val_accu
            best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    trainer.model.load_state_dict(best_model_wts)
    train_loss,train_accu=trainer.test(test_dataset)
    print('---------------test---------------')
    print('test_loss: {:.4f} test_accu: {:.4f}'.format(
                train_loss, train_accu))
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), args.save_model_path)
