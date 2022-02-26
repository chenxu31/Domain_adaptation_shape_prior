#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=5
account=rrg-ebrahimi
save_dir=check_upconv2_disp1_runs

declare -a StringArray=(
# CT2MRI
# lr = 0.00001
#------------------MR2CT
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r1_disp_upconv2_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r2_disp_upconv2_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.0001 Trainer.save_dir=${save_dir}/MRI2CT_301Areg_r3_disp_upconv2_seed3"


#
"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=1 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r1_disp_upconv2_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=2 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r2_disp_upconv2_seed3"

"python main.py seed=123 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_upconv2"
"python main.py seed=231 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_upconv2_seed2"
"python main.py seed=321 Optim.lr=0.00001 DA.source=MRI DA.target=CT Trainer.name=align_IndividualBN DA.batchsize_indicator=6 DA.double_bn=True DA.align_layer.name=Up_conv2 DA.multi_scale=3 DA.displacement=True Scheduler.RegScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/MRI2CT_401Areg_r3_disp_upconv2_seed3"


)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done


