#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser
from config import parse_args



def convert_tree(dep_list, target_idx, target_pos):
    v = "( " + target_pos + " " + str(target_idx) + " )"
    for i in range(len(dep_list)):
        if dep_list[i][3] == -1: #used edge
            continue
        mod_idx = i
        mod_pos = dep_list[i][1]
        head_idx = int(dep_list[i][2])
        head_pos = dep_list[head_idx][1]
        rel = dep_list[i][3]

        if mod_idx == target_idx:
            dep_list[i][3] = -1
            v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

    for i in range(len(dep_list)):
        if i == target_idx:
            break
        if dep_list[i][3] == -1:
            continue
        mod_idx = i
        mod_pos = dep_list[i][1]
        head_idx = int(dep_list[i][2])
        head_pos = dep_list[head_idx][1]
        rel = dep_list[i][3]

        if head_idx == target_idx:
            dep_list[i][3] = -1
            v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

    for i in reversed(range(len(dep_list))):
        if i == target_idx:
            break
        if dep_list[i][3] == -1:
            continue
        mod_idx = i
        mod_pos = dep_list[i][1]
        head_idx = int(dep_list[i][2])
        head_pos = dep_list[head_idx][1]
        rel = dep_list[i][3]

        if head_idx == target_idx:
            dep_list[i][3] = -1
            v = "( " + rel + " " + convert_tree(dep_list, mod_idx, mod_pos) + " " + convert_tree(dep_list, head_idx, head_pos) + " )"

    return v

def convert(company_name,words,postags,arcs):
    arc_head = []
    arc_relation = []
    for arc in arcs:
        head = arc.head - 1
        arc_head.append(head)
        arc_relation.append(arc.relation)
    dep_parsed_sent_list = zip(list(words),list(postags),arc_head,arc_relation)
    #print(dep_parsed_sent_list)
    dep_parsed_sent_list = [list(d) for d in dep_parsed_sent_list]
    #print(dep_parsed_sent_list)

    #find the target
    target_idx = -1
    target_pos = ""
    for i,dep in enumerate(dep_parsed_sent_list):
        if company_name == dep[0]:
            #print (dep[0])
            target_idx = i
            target_pos = dep[1]
        if dep[2] == -1:
            dep[3] = -1
    if target_idx == -1:
        return '###'

    #build tree according to the target
    target_tree = convert_tree(dep_parsed_sent_list, target_idx, target_pos)
    #print('dep_tree:',target_tree)

    #check all the dep have been used
    for dep in dep_parsed_sent_list:
        if dep[3] != -1:
            return '###'

    return target_tree


if __name__ == "__main__":
    args=parse_args()
    LTP_DATA_DIR='../data/ltp_data'

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')#分词
    par_model_path = os.path.join(LTP_DATA_DIR,'parser.model')#语法分析

    segmentor=Segmentor()
    segmentor.load(cws_model_path)

    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    parser=Parser()
    parser.load(par_model_path)

    entity_name = "火车"
    #sentence =[ "专门打电话来问我要不要买手机","最近想买部手机","我想入手一部索尼的手机,主要用于日常拍摄和毕业旅行"]

    #mode_list=['val','test','train']
    trigger_words=['火车','火车站','火车票','车票','动车','高铁','高铁站','动车站','高铁票','动车票']
    mode_list=['val','test','train']
    for mode in mode_list:
        sentence=[]
        domain_data_path='../data/'+args.domain+'/'+args.domain+'_'+mode
        sent_path=args.output_dir+args.domain+'_seged_sent_'+mode
        tree_path=args.output_dir+args.domain+'_s_tree_'+mode

        with open(domain_data_path,'r') as reader:
            for line in reader:
                sentence.append(line)
        with open(sent_path,'w') as writer1:
                with open(tree_path,'w') as writer2:
                    for line in sentence:

                        label= line.strip().split()[0]
                        sen = line.strip().split()[1:]
                        #words = segmentor.segment(' '.join(sen))
                        words = sen
                        tag=-1
                        for tri in trigger_words:
                            if tri in words:
                                entity_name=tri
                                tag=1
                                break
                        if tag==-1:
                            print ' '.join(words)
                            entity_name=words[len(words)/2]

                            #continue#母婴领域里面触发词没法囊括所有的细节，不够搞定，所以打算考虑词性来搞定

                        writer1.write(label+' '+' '.join(words)+'\n')
                        postags = postagger.postag(words)

                        arcs=parser.parse(words,postags)
                        ####################确定entity_name####################需要做实验
                        tree=convert(entity_name,words,postags,arcs)
                        writer2.write(tree+'\n')

    segmentor.release()
    postagger.release()
    parser.release()
