model="resnet18"
dataset="cifar10"
ood_dataset="imagenet_tiny"
query_dataset="cifar100"

epochs=150  # 50 for MNIST, FashionMNIST
lr_step=50  # 20 for MNIST, FashionMNIST

gamma=0.3
delta=1
epsilon=1000

path="../results/${model}/${dataset}/"

# defense training
python yaml_all.py --path ${path} --device "cuda:0" --defense "ini" --attack "knockoff" --knockoff_disable_pbar \
    --ini_gamma ${gamma} --ini_delta ${delta} --ini_epsilon ${epsilon} --ini_use_pcgrad \
    --dataset ${dataset} --ini_dataset_ood ${ood_dataset} --knockoff_sur_dataset ${query_dataset} \
    --ini_epochs ${epochs} --ini_lr_step ${lr_step}
python -u defense_entry.py --config ${path}

attack="knockoff"
pred_type="soft"

path="../results/${model}/${dataset}/${attack}/${pred_type}"
attack_path="../results/${model}/${dataset}/"
# attack
python yaml_all.py --path ${path} --device "cuda:0" --defense "ini" --attack ${attack} --pred_type ${pred_type} --knockoff_disable_pbar \
    --ini_gamma ${gamma} --ini_delta ${delta} --ini_epsilon ${epsilon} --ini_use_pcgrad \
    --dataset ${dataset} --ini_dataset_ood ${ood_dataset} --knockoff_sur_dataset ${query_dataset} \
    --ini_epochs ${epochs} --ini_lr_step ${lr_step} --attack_checkpoint_path ${attack_path}
python -u attack_entry.py --config ${path}