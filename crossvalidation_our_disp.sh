#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0613_disp3_runs

declare -a StringArray=(

#todo disp=1, output layer, resolution=256, ent+align
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_301joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_301joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_301joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_405joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_405joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp3_501Ent_405joint_seed3"

# ---
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_301joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_301joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_301joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_405joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_405joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp3_401Ent_405joint_seed3"

#---
"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_305joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_305joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_305joint_seed3"

"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed1"
"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed2"
"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed3"

)

for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
