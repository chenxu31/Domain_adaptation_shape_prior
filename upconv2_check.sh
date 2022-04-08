#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=rrg-ebrahimi
save_dir=0407_upconv2_checkruns
declare -a StringArray=(

# upconv2 cross correlations
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_205joint_cc"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_205joint_cc_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_205joint_cc_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cc"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cc_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.cc_based=True DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cc_seed3"

# upconv2 projector_clusters:5, 8, 10, 20 disp0
#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster10_seed1"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster10_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster10_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster20_seed1"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster20_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001   Trainer.save_dir=${save_dir}/disp0_501ent_201joint_cluster20_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=8  DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_cluster8_seed1"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=8  DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_cluster8_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=8  DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501ent_305joint_cluster8_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster10_seed1"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster10_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=10 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster10_seed3"

#"python main.py seed=123 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster20_seed1"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster20_seed2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Up_conv2 DA.align_layer.clusters=20 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.0000001   Trainer.save_dir=${save_dir}/disp0_601ent_205joint_cluster20_seed3"

)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
