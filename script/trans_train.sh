#! /bin/bash

d=`date +%m%d`

model_dir=./model_param/script
log_dir=./model_param/script/log/lambda
main_dir=.
mkdir -p ${model_dir}
mkdir -p ${log_dir}

# train flight movie baby phone laptop
s_domain=train
echo source_domain:${s_domain}
t_domain=movie
echo target_domain:${t_domain}
#for i in 0.25 0.5 1.5 2 1.75
for i in 1
do
	python ./trans_main.py \
	--s_train_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_train \
	--s_train_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_train \
	--s_val_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_val \
	--s_val_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_val \
	--t_train_sent ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_seged_sent_train \
	--t_train_tree ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_s_tree_train \
	--t_val_sent ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_seged_sent_val \
	--t_val_tree ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_s_tree_val \
	--s_test_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_test \
	--s_test_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_test \
	--t_test_sent ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_seged_sent_test \
	--t_test_tree ${main_dir}/${t_domain}_tree/prop_relation_split/${t_domain}_s_tree_test \
	--save_model_path ${model_dir}/${s_domain}_${t_domain}${d}${i}.pkl \
	--load_model_path ${model_dir}/${s_domain}.pkl  \
	--beta ${i} \
	>${log_dir}/${s_domain}_${t_domain}${d}${i}.log
done
