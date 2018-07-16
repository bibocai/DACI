#! /bin/bash




for s_domain in baby movie train flight laptop
do
    res_dir=../${s_domain}_tree/prop_relation_split/
    mkdir -p ${res_dir}
    python gen_${s_domain}_tree.py \
    --domain ${s_domain} \
    --output_dir ${res_dir}
done

