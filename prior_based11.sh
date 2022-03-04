#!/usr/bin/env bash

set  -e -u -o pipefail

CC_WRAPPER_PATH="CC_wrapper.sh"

source $CC_WRAPPER_PATH

time=2
account=rrg-ebrahimi
save_dir=priorbased4444

declare -a StringArray=(

#------------------MR2CT
#cluster: prior
#echeduler: ent

"python main.py seed=123 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior"
"python main.py seed=231 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior2"
"python main.py seed=321 Optim.lr=0.00001 Trainer.name=priorbased DA.double_bn=True DA.source=MRI DA.target=CT DA.batchsize_indicator=6 Scheduler.RegScheduler.max_value=0.00001 Scheduler.ClusterScheduler.max_value=0.00001  Trainer.save_dir=${save_dir}/prior_401Ent_401prior3"

)

for cmd in "${StringArray[@]}"
do
	echo ${cmd}
	CC_wrapper "${time}" "${account}" "${cmd}" 16

done

