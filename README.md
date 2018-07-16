dependency：

pytorch : Version 0.2.0 
tqdm : Version 4.19.4
pyltp : Version: 0.1.9.1 

preparation :
1. down the ltp model version 3.3.1:
https://pan.baidu.com/share/link?shareid=1988562907&uk=2738088569#list/path=%2F
'pos.model' 'cws.model' 'parser.model' is needed and Please put the model in path  '/data/ltp_data/'
2. download the embedding data : https://pan.baidu.com/s/1GwX8lSjthxoveRfxB6k5Cw
Please put it in the path "./data/embedding_weibo.data"

the code is runned in three steps:
1. use the script file in fold "generate_tree" to pre-precess the origin data. This step can get the depenency tree presentation of the origin data which is saved in the fold "train_tree","phone_tree" etc.
 command: sh gen_tree.sh
 we've already provided the output of the script in the zip file  
2. run the train_source_model.sh to get the ".pkl" model file while can be used to to predict the origin domain data.  
 command : sh ./script/train_source_model.sh
3. run the trans_train.sh to get the domain transfer answer.
 commdand : sh ./script/trans_train.sh


