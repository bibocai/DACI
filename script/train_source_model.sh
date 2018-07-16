#! /bin/bash

d=`date +%m%d`

model_dir=./model_param/script
log_dir=./model_param/script/log
main_dir=.
mkdir -p ${model_dir}
mkdir -p ${log_dir}

s_domain=laptop
echo ${s_domain}

python ./main.py \
--s_train_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_train \
--s_train_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_train \
--s_val_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_val \
--s_val_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_val \
--s_test_sent ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_seged_sent_test \
--s_test_tree ${main_dir}/${s_domain}_tree/prop_relation_split/${s_domain}_s_tree_test \
--save_model_path ${model_dir}/${s_domain}_${d}.pkl
# --load_model_path ${model_dir}/${domain} \
# &>${log_dir}/${domain}_${d}.log



