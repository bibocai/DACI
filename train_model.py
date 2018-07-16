# -*- coding: utf-8 -*-

from tqdm import tqdm
from config import use_cuda
import torch
from torch.autograd import Variable as Var
from torch.utils.data import  Dataset
# seg_sent : [label  seged words]

class genDataset(Dataset):
    def __init__(self,path_seged_sen,path_s_tree):
        self.seged_sen=[]
        self.label=[]
        self.s_tree=[]
        with open(path_seged_sen,'r') as reader:
            for line in reader:
                words=line.strip().split()
                self.seged_sen.append(words[1:])
                self.label.append(int(words[0]))
        with open(path_s_tree,'r') as reader:
            for line in reader:
                self.s_tree.append(line)
        self.size=len(self.label)
    def __len__(self):
        return self.size
    def __getitem__(self,index):
        return self.seged_sen[index],self.s_tree[index],self.label[index]


class Trainer(object):
    def __init__(self, model, criterion, optimizer,batchsize,args=0):
        super(Trainer, self).__init__()
        # self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0
        self.batchsize  = batchsize
        self.use_cuda= use_cuda
        self.gamma = -6e-4
        self.beta = 3e2
        #self.beta = 0
        if args!=0:
            self.beta = self.beta*float(args.beta)
        print self.beta
    # helper function for training
    #一个dataset

    #add the mmd in the train progress the bandwidth is a exhausting question
    
    def one_iter_mmd(self,x_s1,x_s2,x_t1,x_t2):
        x_ss=torch.exp(torch.pow((x_s1-x_s2),2).sum()*self.gamma)
        x_tt=torch.exp(torch.pow((x_t1-x_t2),2).sum()*self.gamma)
        x_s1t2=torch.exp(torch.pow((x_s1-x_t2),2).sum()*self.gamma)
        x_s2t1=torch.exp(torch.pow((x_s2-x_t1),2).sum()*self.gamma)
        mmd_dis=(     x_ss
                    + x_tt
                    - x_s1t2
                    - x_s2t1)
        return mmd_dis

    def trans_train(self,source_dataset,target_dataset,val_dataset):
        self.model.train()
        if self.use_cuda:
            self.model.cuda()
        self.optimizer.zero_grad()
        total_loss = 0.0
        classify_loss = 0.0
        mmd_loss = 0.0
        distance = 0.0
        total_dis = 0.0
        batch_mmd_loss = 0.0
        batch_distance=0.0
        batch_total_loss= 0.0
        s_indices = torch.randperm(len(source_dataset)/2*2)#要求必须是偶数
        #注意对这里做一些修改 考虑选取的对象的长度可能尽量一致会比较好
        #这部分可能做数据预处理让句子长度尽量相近会比较好
        target_num = len(target_dataset)
        batch_classify_loss=0.0


        for idx in tqdm(range(len(s_indices)/2),desc='Training epoch'):
            s1_seged_sen,s1_tree,s1_label=source_dataset[s_indices[2*idx]]
            s2_seged_sen,s2_tree,s2_label=source_dataset[s_indices[2*idx+1]]

            t1_seged_sen,t1_tree,t1_label=target_dataset[s_indices[2*idx]%target_num]
            t2_seged_sen,t2_tree,t2_label=target_dataset[s_indices[2*idx+1]%target_num]

            # print 's1_segs_sen,s1_tree,s1_label',' '.join(s1_seged_sen),s1_tree,s1_label
            # print 't1_segs_sen,t1_tree,t1_label',' '.join(t1_seged_sen),t1_tree,t1_label
            x_s1,y_s1=self.model(s1_seged_sen,s1_tree,trans=True)
            # print x_s1,y_s1
            x_s2,y_s2=self.model(s2_seged_sen,s2_tree,trans=True)
            x_t1,y_t1=self.model(t1_seged_sen,t1_tree,trans=True)
            x_t2,y_t2=self.model(t2_seged_sen,t2_tree,trans=True)

            #print x_s1,x_s2,s1_label,s2_label
            s1_label=Var(torch.LongTensor([s1_label]))
            s2_label=Var(torch.LongTensor([s2_label]))
            if self.use_cuda:
                s1_label=s1_label.cuda()
                s2_label=s2_label.cuda()

            classify_loss = self.criterion(torch.cat((y_s1.view(1,-1),y_s2.view(1,-1)),0),torch.cat((s1_label,s2_label)))

            #计算distance ,得到动态变化的bandwidth
            d_ss=torch.pow((x_s1.data-x_s2.data),2).sum()
            d_tt=torch.pow((x_t1.data-x_t2.data),2).sum()
            d_s1t2=torch.pow((x_s1.data-x_t2.data),2).sum()
            d_s2t1=torch.pow((x_s2.data-x_t1.data),2).sum()
            distance = d_ss + d_s2t1 + d_s1t2 + d_tt
            batch_distance += distance

            mmd_loss = self.one_iter_mmd(x_s1,x_s2,x_t1,x_t2)
            add_loss = classify_loss+self.beta*mmd_loss
            #print classify_loss,mmd_loss.data[0]
            batch_classify_loss += classify_loss.data[0]
            batch_mmd_loss += mmd_loss.data[0]
            total_loss += add_loss.data[0]
            batch_total_loss += add_loss.data[0]
            add_loss.backward()
####

            if idx % self.batchsize == 0 and idx > 0:
                #print 'batch_distance  = ' ,batch_distance
                self.gamma = -1/(batch_distance)
                print 'gamma:', self.gamma
                batch_distance = 0
                self.optimizer.step()
                self.optimizer.zero_grad()

                ##输出一下每个batch的mmd的距离，看看mmd距离是不是有明显的下降

                print 'batch_mmd_loss',batch_mmd_loss
                print 'batch_classify_loss',batch_classify_loss
                print 'batch_total_loss',batch_total_loss
                batch_mmd_loss=0.0
                batch_total_loss =0.0
                batch_classify_loss=0.0
                val_loss,val_accu=self.test(val_dataset)
                print 'during training after one batch the accu'
                print 'after one batch,val_acc',val_accu
            self.epoch+=1


        return total_loss / len(source_dataset)
            #这样就不能动态的自适应的调mmd的bandwidth了

    def train(self, dataset):
        self.model.train()
        if self.use_cuda:
            self.model.cuda()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset))
        for idx in tqdm(range(len(dataset)),desc='Training epoch ' + str(self.epoch + 1) + ''):
            seged_sen,s_tree,label = dataset[indices[idx]]
            # target = Var(map_label_to_target(label, dataset.num_classes))
            label = Var(torch.LongTensor([label]))
            if self.use_cuda:
            #     linput, rinput = linput.cuda(), rinput.cuda()
                 label = label.cuda()
            output = self.model(seged_sen,s_tree)
            loss = self.criterion(output.view(1,-1), label)

            total_loss += loss.data[0]
            loss.backward()
            if idx % self.batchsize == 0 and idx > 0:
                self.optimizer.step() #注意设置learning rate的时候解决跟batch_size相关的问题
                self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    #在整个训练集上做eval
    def test(self, dataset):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        predictions = torch.zeros(len(dataset))
        # indices = torch.arange(1, dataset.num_classes + 1)
        for idx in tqdm(range(len(dataset)),desc='Testing epoch  ' + str(self.epoch) + ''):
            seged_sen, s_tree, label = dataset[idx]
            # linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
            label = Var(torch.LongTensor([label]), volatile=True)
            if self.use_cuda:
                # linput, rinput = linput.cuda(), rinput.cuda()
                label = label.cuda()
            output = self.model(seged_sen,s_tree)
            loss = self.criterion(output.view(1,-1), label)
            total_loss += loss.data[0]
            output = output.data.view(1,-1).cpu()

            # predictions[idx] = torch.dot(indices, torch.exp(output))
            _ , pred = torch.max(output,1)

            correct += torch.sum((label.data.cpu() == pred ))
            predictions[idx]=pred[0]
#           type(correct):torch.ByteTensor of size 1
        print correct
        return total_loss / len(dataset), float(correct)/len(dataset)

if __name__ == '__main__':
    train_dataset = genDataset('./train_seged_sent','./train_s_tree')
    print train_dataset[0]



#以下的内容将会是通用的用在main函数中的语句，详细可以看main函数
    # 实例化train object
#    since=time.time()
#    best_acc=0.0
#    best_model_wts = model.state_dict()
#    for epoch in range(100):
#        train_loss             = trainer.train(train_dataset)
#        train_loss, train_accu = trainer.test(train_dataset)
#        val_loss ,val_accu    = trainer.test(val_dataset)
        # test_loss, test_pred   = trainer.test(test_dataset)

#        print('---------------train---------------')
#        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                train_loss, train_accu))
#        print('--------------val------------------')
#        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                val_loss, val_accu))
#
#            # deep copy the model
#        if val_acc > best_acc:
#            best_acc = val_acc
#            best_model_wts = model.state_dict()
#
#    time_elapsed = time.time() - since
#    print('Training complete in {:.0f}m {:.0f}s'.format(
#        time_elapsed // 60, time_elapsed % 60))
#    print('Best val Acc: {:4f}'.format(best_acc))
#
#    # load best model weights
#    model.load_state_dict(best_model_wts)

#    torch.save(model.static_dict(), "./train_status.pkl")
