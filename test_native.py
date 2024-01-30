import warnings
import torch
from imageio import imsave
from torch import Tensor
from typing import Union, Iterable

from arch.DomainSpecificBNUnet import switch_bn as _switch_bn, convert2TwinBN
from arch.unet import UNet
from demo.criterions import nullcontext
from pathlib import Path
import argparse
import numpy as np
import numpy
import os
import sys
import pdb
import platform

if platform.system() == 'Windows':
    UTIL_DIR = r"E:\我的坚果云\sourcecode\python\util"
else:
    UTIL_DIR = r"/home/chenxu/我的坚果云/sourcecode/python/util"

sys.path.append(UTIL_DIR)
import common_metrics
import common_net_pt as common_net
import common_pelvic_pt as common_pelvic


torch.backends.cudnn.benchmark = True


def produce(switch_bn, model, x):
    with switch_bn(model, 1):
        pred = model(x)

    return pred.softmax(1).unsqueeze(2)


def main(device, args):
    if args.task == 'pelvic':
        common_file = common_pelvic
        _, test_data_t, _, test_label_t = common_pelvic.load_test_data(args.data_dir)
    else:
        raise NotImplementedError(args.task)

    num_classes = common_file.NUM_CLASSES
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    double_bn = True
    Smodel = UNet(num_classes=num_classes, input_dim=1)
    if double_bn:
        Smodel = convert2TwinBN(Smodel)
    weight = os.path.join(args.checkpoint_dir, "%s.pth" % args.pretrained_tag)
    state_dict = torch.load(weight, map_location=torch.device('cpu'))
    Smodel.load_state_dict(state_dict.get('model'))
    Smodel = Smodel.eval()
    Smodel.to(device)

    switch_bn = _switch_bn if True else nullcontext

    patch_shape = (1, test_data_t.shape[2], test_data_t.shape[3])
    test_t_dsc = numpy.zeros((test_data_t.shape[0], num_classes - 1), numpy.float32)
    test_t_assd = numpy.zeros((test_data_t.shape[0], num_classes - 1), numpy.float32)
    with torch.no_grad():
        for i in range(test_data_t.shape[0]):
            pred = common_net.produce_results(device, lambda x: produce(switch_bn, Smodel, x), [patch_shape, ],
                                              [test_data_t[i], ], data_shape=test_data_t[i].shape,
                                              patch_shape=patch_shape, is_seg=True, num_classes=num_classes)
            pred = pred.argmax(0).astype(numpy.float32)
            test_t_dsc[i] = common_metrics.calc_multi_dice(pred, test_label_t[i], num_cls=num_classes)
            test_t_assd[i] = common_metrics.calc_multi_assd(pred, test_label_t[i], num_cls=num_classes)

            if args.output_dir:
                common_file.save_nii(pred, os.path.join(args.output_dir, "syn_%d.nii.gz" % i))

    msg = "test_t_dsc:%f/%f  test_t_assd:%f/%f" % \
          (test_t_dsc.mean(), test_t_dsc.std(), test_t_assd.mean(), test_t_assd.std())
    print(msg)

    if args.output_dir:
        with open(os.path.join(args.output_dir, "result.txt"), "w") as f:
            f.write(msg)

        numpy.save(os.path.join(args.output_dir, "test_t_dsc.npy"), test_t_dsc)
        numpy.save(os.path.join(args.output_dir, "test_t_assd.npy"), test_t_assd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=int, default=0, help="gpu device id")
    parser.add_argument('--data_dir', type=str, default='/home/chenxu/datasets/pelvic/h5_data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint_dir')
    parser.add_argument('--pretrained_tag', type=str, default='best', choices=["best", "last"], help='task')
    parser.add_argument('--output_dir', type=str, default='', help='output dir')
    parser.add_argument('--task', type=str, default='pelvic', choices=["pelvic", ], help='task')
    args = parser.parse_args()

    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device = torch.device("cpu")

    main(device, args)
