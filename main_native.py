import argparse
import os
import sys
import pdb
import torch

from arch.DomainSpecificBNUnet import convert2TwinBN, switch_bn as _switch_bn
from configure import ConfigManager
from demo.criterions import nullcontext
from scheduler.customized_scheduler import RampScheduler
from scheduler.warmup_scheduler import GradualWarmupScheduler
from trainers.align_IBN_trainer import align_IBNtrainer_native
from trainers.entropy_DA_trainer import EntropyDA
from utils.radam import RAdam
from utils.utils import fix_all_seed_within_context, fix_all_seed
from arch.unet import UNet
import platform

if platform.system() == 'Windows':
    NUM_WORKERS = 0
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    NUM_WORKERS = 4
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_pelvic_pt as common_pelvic


class DatasetPelvic(common_pelvic.Dataset):
    def __getitem__(self, idx):
        ret = super(DatasetPelvic, self).__getitem__(idx)

        return [ret["image"], ret["label"] if "label" in ret else ret["image"]], ""


def main(args):
    cmanager = ConfigManager("configs/config.yaml", strict=True)
    config = cmanager.config
    config["Data_input"]["dataset"] = args.task
    config["Data_input"]["data_dir"] = args.data_dir
    config["Data_input"]["num_class"] = args.num_classes
    config["DataLoader"]["batch_size"] = args.batch_size
    config["Trainer"]["checkpoint_dir"] = args.checkpoint_dir
    config["Optim"]["lr"] = args.lr
    fix_all_seed(config['seed'])
    switch_bn = _switch_bn if config['DA']['double_bn'] else nullcontext

    with fix_all_seed_within_context(config['seed']):
        model = UNet(num_classes=config['Data_input']['num_class'], input_dim=1)
        # model = Enet(num_classes=config['Data_input']['num_class'], input_dim=1)

    with fix_all_seed_within_context(config['seed']):
        if config['DA']['double_bn']:
            model = convert2TwinBN(model)
        optimizer = RAdam(model.parameters(), lr=config["Optim"]["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(90, 1), eta_min=1e-7)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=300, total_epoch=10, after_scheduler=scheduler)

    if config['Data_input']['dataset'] == 'pelvic':
        dataset_s = DatasetPelvic(args.data_dir, "ct", debug=args.debug)
        dataset_t = DatasetPelvic(args.data_dir, "cbct", debug=args.debug)
    else:
        raise NotImplementedError(config['Data_input']['dataset'])

    dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=NUM_WORKERS)
    dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=NUM_WORKERS)

    RegScheduler = RampScheduler(**config['Scheduler']["RegScheduler"])
    weight_cluster = RampScheduler(**config['Scheduler']["ClusterScheduler"])

    trainer = align_IBNtrainer_native(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        TrainS_loader=dataloader_s,
        TrainT_loader=dataloader_t,
        weight_scheduler=RegScheduler,
        weight_cluster=weight_cluster,
        switch_bn=switch_bn,
        config=config,
        num_batches=min(len(dataloader_s), len(dataloader_t)),
        **config['Trainer']
    )

    # trainer.inference(identifier='last.pth')
    trainer.start_training()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default='/home/chenxu/datasets/pelvic/h5_data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint_dir')
    parser.add_argument('--task', type=str, default='pelvic', choices=["pelvic", ], help='task')
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--debug', type=int, default=0, help='debug flag')
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        #device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #device = torch.device("cpu")

    main(args)
