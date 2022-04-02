#!/usr/bin/env bash

set -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=3
account=rrg-ebrahimi
save_dir=0401_prostate_baseline
declare -a StringArray=(

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=3 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=8 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline56_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/mise2state_lower_baseline63_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=3 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=8 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline56_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=promise DA.target=prostate DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/mise2state_upper_baseline63_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=3 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=8 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline56_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=baseline Trainer.save_dir=${save_dir}/state2mise_lower_baseline63_seed1"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=3 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline21_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=8 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline56_seed1"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 DA.double_bn=False DA.source=prostate DA.target=promise DA.batchsize_indicator=9 Trainer.name=upperbaseline Trainer.save_dir=${save_dir}/state2mise_upper_baseline63_seed1"

#align
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=promise DA.target=prostate DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/mise2state_disp0_401Ent_301regjoint63"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=promise DA.target=prostate DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/mise2state_disp1_401Ent_301regjoint63"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=promise DA.target=prostate DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/mise2state_disp3_401Ent_301regjoint63"

"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=False Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/state2mise_disp0_401Ent_301regjoint63"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=1 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/state2mise_disp1_401Ent_301regjoint63"
"python main.py seed=123 Optim.lr=0.00001 Data_input.dataset=prostate Data_input.num_class=2 Trainer.name=align_IndividualBN DA.double_bn=True DA.source=prostate DA.target=promise DA.batchsize_indicator=9 DA.statistic=False DA.align_layer.name=Deconv_1x1 DA.multi_scale=1 DA.displacement=True DA.displace_scale=3 Scheduler.RegScheduler.max_value=0.0001 Scheduler.ClusterScheduler.max_value=0.00001 Trainer.save_dir=${save_dir}/state2mise_disp3_401Ent_301regjoint63"


)


for cmd in "${StringArray[@]}"; do
  echo ${cmd}
  CC_wrapper "${time}" "${account}" "${cmd}" 16

done
