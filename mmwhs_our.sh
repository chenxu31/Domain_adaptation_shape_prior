#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=4
account=def-chdesa
save_dir=mmwhs_our_upconv2_resolutions
declare -a StringArray=(
# todo: cross-validation for tesing the alignment loss weight (Table1)
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_305joint_fold1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_305joint_fold2"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=3 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_305joint_fold3"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=4 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0005 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_305joint_fold4"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_301joint_fold1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_301joint_fold2"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=3 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_301joint_fold3"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=4 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp0_405Ent_301joint_fold4"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold2"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=3 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold3"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=4 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_fold4"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold2"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=3 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold3"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=4 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_401joint_fold4"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=1 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_505joint_fold1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_505joint_fold2"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=3 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_505joint_fold3"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=4 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.000005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/disp0_501Ent_505joint_fold4"
#
##todo: displacement(0,1,3,5,13)
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp0_401Ent_405joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1_401Ent_301joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp3_405Ent_301joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp5_401Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp5_401Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp5_401Ent_301joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp13_405Ent_301joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp13_405Ent_301joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00005 Trainer.save_dir=${save_dir}/disp13_405Ent_301joint_seed3"
#
## todo: upconv2 displacement(cross-correlation)
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc_501Ent_201joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc_501Ent_201joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_cc_501Ent_201joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp1_501Ent_205joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp1_501Ent_205joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp1_501Ent_205joint_seed3"

#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp3_501Ent_205joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp3_501Ent_205joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp3_501Ent_205joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_201joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_201joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=5 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_201joint_seed3"
#
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_205joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_205joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True DA.displace_scale=13 DA.align_layer.cc_based=True Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.000001 Trainer.save_dir=${save_dir}/upconv2_ccdisp5_501Ent_205joint_seed3"

## todo: upconv2 clusters
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=5 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k5_401Ent_205joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=8 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k8_401Ent_205joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=10 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k10_401Ent_205joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=15 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k15_401Ent_205joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=20 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k20_401Ent_205joint_seed1"
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=align_IndividualBN DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=False DA.align_layer.clusters=30 DA.align_layer.cc_based=False Scheduler.RegScheduler.max_value=0.005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/upconv2_k30_401Ent_205joint_seed1"
#
## todo : both layer
#"python main.py seed=10 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_401Ent_205joint_seed1"
#"python main.py seed=20 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_401Ent_205joint_seed2"
#"python main.py seed=30 Optim.lr=0.00001 Data_input.dataset=mmwhs Data.kfold=0 Trainer.name=combinationlayer DA.double_bn=True DA.batchsize_indicator=9 DA.statistic=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True DA.align_layer.cc_based=True DA.weight1=200 Scheduler.RegScheduler.max_value=0.00005 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/disp1r3_200bothlayer_401Ent_205joint_seed3"



)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
