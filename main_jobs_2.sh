#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=main_jobs_alignweight

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
# Deconv_1x1, Areg=1, 0.1, 0.01, 0.001
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=1     Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_0disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=0.1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_0disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_0disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=False Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_0disp_prediction"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=1     Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=0.1   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=0.01  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_disp_prediction"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.name=Deconv_1x1 DA.displacement=True Scheduler.RegScheduler.max_value=0.001 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_201Areg_disp_prediction"

# Up_conv2, K=5, 10, 20, 40, Areg
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_0disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_0disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_0disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_0disp_upconv2_5"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_disp_upconv2_5"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=5 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Are_disp_upconv2_5"

# 10
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_0disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_0disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_0disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_0disp_upconv2_10"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_disp_upconv2_10"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=10 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_disp_upconv2_10"

# 20
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_0disp_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_0disp_20c"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_disp_upconv2_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_disp_upconv2_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_disp_upconv2_20c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=20 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_disp_upconv2_20c"

#
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_0disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_0disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_0disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=False Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_0disp_upconv2_40c"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=10   Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_10Areg_disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=1    Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_1Areg_disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.1  Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_01Areg_disp_upconv2_40c"
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.double_bn=True DA.align_layer.clusters=40 DA.align_layer.name=Up_conv2 DA.displacement=True Scheduler.RegScheduler.max_value=0.01 Scheduler.ClusterScheduler.max_value=0 Trainer.save_dir=${save_dir}/MRI2CT_101Areg_disp_upconv2_40c"



)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


