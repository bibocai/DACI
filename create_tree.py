# coding: utf-8
import os

# from pyltp import SentenceSplitter
from pyltp import Postagger
from pyltp import Segmentor
from pyltp import Parser

class TreeNode(object):
    def __init__(self,k=None,l=None,r=None):
        self.key = k
        self.left = l
        self.right = r

def create_tree(s):
    #print(s)
    s = s.strip()[2:-2].split(' ')
    #print(s)
    root = TreeNode()
    if len(s) == 2:
        root.key = s[1].strip()
        return root
    root.key = s[0].strip()
    flag = 1
    for i in range(2,len(s)):
        if s[i] == '(':
            flag += 1
        if s[i] == ')':
            flag -= 1
        if flag == 0:
            root.left = create_tree(' '.join(s[1:i+1]))
            root.right = create_tree(' '.join(s[i+1:]))
            break
    return root



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


def show_tree(s_tree,depth):
    if not s_tree:
        return
    print(' '*depth*5+s_tree.key)
    show_tree(s_tree.left,depth+1)
    show_tree(s_tree.right,depth+1)
    return

if __name__ == '__main__':
    LTP_DATA_DIR='/data/ltp/ltp-models/3.3.1/ltp_data'

    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')#分词
    par_model_path = os.path.join(LTP_DATA_DIR,'parser.model')#语法分析

    segmentor=Segmentor()
    segmentor.load(cws_model_path)

    postagger = Postagger() # 初始化实例
    postagger.load(pos_model_path)  # 加载模型

    parser=Parser()
    parser.load(par_model_path)

    entity_name = "手机"
    #sentence =[ "专门打电话来问我要不要买手机","最近想买部手机","我想入手一部索尼的手机,主要用于日常拍摄和毕业旅行"]

    #mode_list=['val','test','train']
    mode_list=['val','test','train']
    trigger_words=['手机','笔记本','iphone','品牌','水果','小米','三星','索尼','中兴','i5','海尔','诺基亚','联想','安卓','htc','oppo','魅族','黑莓','华为','罗永浩','vivo','锤子','nokia','sony','海信','华硕','康佳','s4','note3','apple','galaxy','i6','huawei','s3','lumia','samsung','酷派','tcl','长虹','i4','note2','4s','酷睿','比亚迪','blackberry','创维','meizu']
    # for mode in mode_list:
    sentence=['想 买 台 笔记本 不要 太 贵','看来 要 买 手机 了 求 推荐']
    

    for line in sentence:
        #words = segmentor.segment(sen)
        # label= line.strip().split()[0]
        words = line.strip().split()
        tag=-1
        for tri in trigger_words:
            if tri in words:
                entity_name=tri
                tag=1
                break
        if tag==-1:
            continue#跳过句子里面没有手机的触发词语的情况
        # writer1.write(label+' '+' '.join(words)+'\n')
        postags = postagger.postag(words)

        arcs=parser.parse(words,postags)
        ####################确定entity_name####################需要做实验
        tree=convert(entity_name,words,postags,arcs)
        s_tree = create_tree(tree)
        show_tree(s_tree,0)

        print('\n')
    segmentor.release()
    postagger.release()
    parser.release()

# if __name__ == '__main__':
#     s = "( ATT ( ni 0 ) ( SBV ( ATT ( ATT ( v 1 ) ( n 2 ) ) ( ATT ( n 3 ) ( COO ( LAD ( c 5 ) ( n 6 ) ) ( n 4 ) ) ) ) ( ADV ( POB ( ATT ( d 8 ) ( ATT ( v 9 ) ( n 10 ) ) ) ( d 7 ) ) ( VOB ( n 14 ) ( RAD ( u 13 ) ( VOB ( n 12 ) ( v 11 ) ) ) ) ) ) )"
#     s_tree = create_tree(s)
#     show_tree(s_tree)
