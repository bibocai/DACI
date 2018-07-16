#-*- coding:utf-8 -*-

from train_model import Trainer,genDataset
import time
from test_pytorch_lstm import TreeLstm
import torch
from config import parse_args

input_size=100
hidden_size=200
batch_size=256

if __name__ == '__main__':
    args=parse_args()
    model = TreeLstm(input_size,hidden_size)
    # model.load_state_dict(torch.load("./model_param/ir-tree-movie-0.01.pkl"))
    model.load_state_dict(torch.load(args.load_model_path))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
    trainer = Trainer(model,criterion,optimizer,batch_size,args)
    s_train_dataset = genDataset(args.s_train_sent,args.s_train_tree)
    s_val_dataset = genDataset(args.s_val_sent,args.s_val_tree)
    s_test_dataset = genDataset(args.s_test_sent,args.s_test_tree)

    t_train_dataset = genDataset(args.t_train_sent,args.t_train_tree)
    t_val_dataset = genDataset(args.t_val_sent,args.t_val_tree)
    t_test_dataset = genDataset(args.t_test_sent,args.t_test_tree)

    since=time.time()
    best_acc=0.0
    best_model_wts = model.state_dict()
    min_loss=0.0
    list_accu=[]
    list_train_loss=[]

    source_loss,source_accu = trainer.test(s_test_dataset)
    print 'source_accu',source_accu
    target_loss,target_accu = trainer.test(t_test_dataset)
    print 'without_mmd_target_accu',target_accu
    for epoch in range(5):

        train_loss = trainer.trans_train(s_train_dataset,t_train_dataset,t_val_dataset)
        val_loss, val_accu = trainer.test(t_val_dataset)

        print('--------------val------------------')
        print('ValLoss: {:.4f} ValAcc: {:.4f}'.format(
              val_loss, val_accu))
        if val_accu > best_acc:
            best_acc = val_accu
        if train_loss <min_loss:
            best_model_wts = model.state_dict()
        list_accu.append(val_accu)
        list_train_loss.append(train_loss)

        print('---------------train-----------------')
        print('TransferTrainLoss: {:.4f}'.format(train_loss))

        print 'source_accu',source_accu
        print 'without_mmd_val_accu',target_accu
        print 'accu',list_accu
        print 'train_loss',list_train_loss
        tmp_loss = list_train_loss
        tmp_acc = list_accu

        print ('min_loss{:.4f},related accu{:.4f}'.format(min(tmp_loss),tmp_acc[tmp_loss.index(min(tmp_loss))]))
        print ('best_accu',max(tmp_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    trainer.model.load_state_dict(best_model_wts)
    test_loss,test_accu = trainer.test(t_test_dataset)
    print('------------------test----------------')
    print('test_accu',test_accu)
    torch.save(model.state_dict(), args.save_model_path)

