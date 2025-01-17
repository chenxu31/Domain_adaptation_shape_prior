import shutil
import warnings
from pathlib import Path
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from torch import Tensor
from typing import Union, Iterable
import torch
from meters.SummaryWriter import get_tb_writer
from utils.utils import switch_plt_backend
from skimage.io import imsave
import os
#from MulticoreTSNE import MulticoreTSNE as TSNE

def save_joint_distribution(images: np.ndarray, root, mode, iter):
    assert images.ndim == 2
    save_path = Path(root, f'joints/{mode}_{iter}').with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imshow(images)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close(plt.gcf())


def plot_joint_matrix(joint: Tensor):
    assert joint.dim() == 4, joint.shape
    n1, n2 = joint.shape[0:2]
    fig = plt.figure()
    # fig = fig.add_subplot(n1, n2)
    joint = joint.detach().cpu().float().numpy()
    for i1 in range(1, n1 + 1):
        for i2 in range(1, n2 + 1):
            ax = plt.subplot(n1, n2, (i1 - 1) * n1 + i2)
            img = joint[i1 - 1, i2 - 1]
            im_ = ax.imshow(img)
            fig.colorbar(im_, ax=ax, orientation='vertical')
    return fig

def plot_joint_matrix1(joint: Tensor, indicator="error"):
    if joint.dim() == 2:
        joint = joint.unsqueeze(0).unsqueeze(0)
    assert joint.dim() == 4, joint.shape
    # n1, n2 = joint.shape[0:2]
    fig = plt.figure()
    # fig = fig.add_subplot(n1, n2)
    joint = joint.detach().squeeze(0).squeeze(0).cpu().float().numpy()
    plt.imshow(joint)
    plt.colorbar(orientation='vertical')
    if indicator == "percenterror":
       plt.clim(0,1)
    elif indicator == "error":
        plt.clim(0, 0.01)
    return fig

def plot_seg(img, label):
    # img_volume = img.squeeze(0)
    fig = plt.figure()
    plt.title(f'{img}')
    # img_volume = tensor2plotable(img_volume)
    # plt.imshow(img_volume, cmap="gray")
    gt_volume = tensor2plotable(label)
    # con = plt.contour(gt_volume)
    plt.imshow(gt_volume, alpha=1, cmap="viridis")
    # plt.show(block=False)
    return fig

def plot_feature(img):
    img_volume = img
    fig = plt.figure()
    img_volume = tensor2plotable(img_volume)
    plt.imshow(img_volume, cmap="gray")
    return fig

def tensor2plotable(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise TypeError(f"tensor should be an instance of Tensor, given {type(tensor)}")


def create_ade20k_label_colormap():
    """Creates a label colormap used in ADE20K segmentation benchmark.

    Returns:
      A colormap for visualizing segmentation results.
    """
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])

def label2colored_image(label: np.ndarray):
    color_map = create_ade20k_label_colormap()
    return torch.from_numpy(color_map[np.asarray(label).astype(int)])

class FeatureMapSaver:

    def __init__(self, save_dir: Union[str, Path], folder_name="vis", use_tensorboard: bool = True) -> None:
        assert Path(save_dir).exists() and Path(save_dir).is_dir(), save_dir
        self.save_dir: Path = Path(save_dir)
        self.folder_name = folder_name
        (self.save_dir / self.folder_name).mkdir(exist_ok=True, parents=True)
        self.use_tensorboard = use_tensorboard

    @switch_plt_backend(env="agg")
    def save_map(self, *, imageS: Tensor, imageT: Tensor, feature_mapS: Tensor, feature_mapT: Tensor, feature_type="feature",
                 cur_epoch: int,
                 cur_batch_num: int, save_name: str) -> None:
        """
        Args:
            image: image tensor with bchw dimension, where c should be 1.
            feature_map1: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_map2: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_type: image or feature. image is going to treat as image, feature would take the argmax on c.
            cur_epoch: current epoch
            cur_batch_num: cur_batch_num
            save_name: the png that would be saved under "save_name_cur_epoch_cur_batch_num.png" in to self.folder_name
                    folder.
        """
        assert feature_type in ("image", "feature")
        assert imageS.dim() == 4, f"image should have bchw dimensions, given {imageS.shape}."
        assert imageT.dim() == 4, f"image should have bchw dimensions, given {imageT.shape}."

        batch_size = feature_mapS.shape[0]
        imageS = imageS.detach()[:, 0].float().cpu()
        imageT = imageT.detach()[:, 0].float().cpu()

        assert feature_mapS.dim() == 4, f"feature_map should have bchw dimensions, given {feature_mapS.shape}."
        if feature_type == "image":
            feature_mapS = feature_mapS.detach()[:, 0].float().cpu()
        else:
            feature_mapS = feature_mapS.max(1)[1].cpu().float()
            feature_mapS = label2colored_image(feature_mapS)

        assert feature_mapT.dim() == 4, f"feature_map should have bchw dimensions, given {feature_mapT.shape}."
        if feature_type == "image":
            feature_mapT = feature_mapT.detach()[:, 0].float().cpu()
        else:
            feature_mapT = feature_mapT.max(1)[1].cpu().float()
            feature_mapT = label2colored_image(feature_mapT)

        for i, (imgS, imgT, f_mapS, f_mapT) in enumerate(zip(imageS, imageT, feature_mapS, feature_mapT)):
            save_path = self.save_dir / self.folder_name / \
                        f"{save_name}_{cur_epoch:03d}_{cur_batch_num:02d}_{i:03d}.png"
            fig = plt.figure(figsize=(3, 3))
            plt.subplot(221)
            plt.imshow(imgS, cmap="gray")
            plt.axis('off')
            plt.subplot(222)
            plt.imshow(imgT, cmap="gray")
            plt.axis('off')
            plt.subplot(223)
            plt.imshow(f_mapS, cmap="gray" if feature_type == "image" else None)
            plt.axis('off')
            plt.subplot(224)
            plt.imshow(f_mapT, cmap="gray" if feature_type == "image" else None)
            plt.axis('off')
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_figure(
                    tag=f"{self.folder_name}/{save_name}_{cur_batch_num * batch_size + i:02d}",
                    figure=plt.gcf(), global_step=cur_epoch, close=True
                )
            plt.close(fig)

    def zip(self) -> None:
        """
        Put all image folders as a zip file, in order to avoid IO things when downloading.
        """
        try:
            shutil.make_archive(str(self.save_dir / self.folder_name.replace("/", "_")), 'zip',
                                str(self.save_dir / self.folder_name))
            shutil.rmtree(str(self.save_dir / self.folder_name))
        except (FileNotFoundError, OSError, IOError) as e:
            logger.opt(exception=True, depth=1).warning(e)

    @property
    @lru_cache()
    def tb_writer(self):
        try:
            writer = get_tb_writer()
        except RuntimeError:
            writer = None
        return writer

class CycleVisualSaver:

    def __init__(self, save_dir: Union[str, Path], folder_name, use_tensorboard: bool = True) -> None:
        assert Path(save_dir).exists() and Path(save_dir).is_dir(), save_dir
        self.save_dir: Path = Path(save_dir)
        self.folder_name = folder_name
        (self.save_dir / self.folder_name).mkdir(exist_ok=True, parents=True)
        self.use_tensorboard = use_tensorboard

    @switch_plt_backend(env="agg")
    def save_map(self, *, real_image: Tensor, fake_image: Tensor, recover_image: Tensor, cur_epoch: int,
                 cur_batch_num: int, save_name: str) -> None:
        """
        Args:
            image: image tensor with bchw dimension, where c should be 1.
            feature_map1: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_map2: tensor with bchw dimension. It would transform to bhw with argmax on c dimension.
            feature_type: image or feature. image is going to treat as image, feature would take the argmax on c.
            cur_epoch: current epoch
            cur_batch_num: cur_batch_num
            save_name: the png that would be saved under "save_name_cur_epoch_cur_batch_num.png" in to self.folder_name
                    folder.
        """
        assert real_image.dim() == 4, f"image should have bchw dimensions, given {real_image.shape}."
        assert fake_image.dim() == 4, f"image should have bchw dimensions, given {fake_image.shape}."
        batch_size = real_image[1].unsqueeze(0).shape[0]

        real_image = real_image[1].detach().float().cpu()
        fake_image = fake_image[1].detach().float().cpu()

        assert recover_image.dim() == 4, f"feature_map should have bchw dimensions, given {recover_image.shape}."
        recover_image = recover_image[1].detach().float().cpu()


        for i, (img_real, img_fake, img_recover) in enumerate(zip(real_image, fake_image, recover_image)):
            save_path = self.save_dir / self.folder_name / \
                        f"{save_name}_{cur_epoch:03d}_{cur_batch_num:02d}_{i:03d}.png"
            fig = plt.figure(figsize=(3, 3))
            plt.subplot(131)
            plt.imshow(img_real, cmap="gray")
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(img_fake, cmap="gray")
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(img_recover, cmap="gray")
            plt.axis('off')
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            if self.use_tensorboard and self.tb_writer is not None:
                self.tb_writer.add_figure(
                    tag=f"{self.folder_name}/{save_name}_{cur_batch_num * batch_size + i:02d}",
                    figure=plt.gcf(), global_step=cur_epoch, close=True
                )
            plt.close(fig)

    def zip(self) -> None:
        """
        Put all image folders as a zip file, in order to avoid IO things when downloading.
        """
        try:
            shutil.make_archive(str(self.save_dir / self.folder_name.replace("/", "_")), 'zip',
                                str(self.save_dir / self.folder_name))
            shutil.rmtree(str(self.save_dir / self.folder_name))
        except (FileNotFoundError, OSError, IOError) as e:
            logger.opt(exception=True, depth=1).warning(e)

    @property
    @lru_cache()
    def tb_writer(self):
        try:
            writer = get_tb_writer()
        except RuntimeError:
            writer = None
        return writer

def save_images(segs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, iter: int) -> None:
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            save_path = Path(root, mode, f'{name}_{iter}').with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_path), seg.detach().numpy())


def draw_pictures(input_data, input_target, save_name, show_legend=False):
    pass
    tsne = TSNE(n_jobs=8, n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(input_data, None)
    label = ["Background", "LVM", "LAC", "LVC", "AA"]
    colors = ["tab:gray", 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    plt.figure(figsize=(4, 4))
    for i, (l, c) in enumerate(zip(label, colors)):
        index = input_target == i
        plt.scatter(*tsne_results[index].transpose(), c=c, label=l, linewidth=0)
    if show_legend:
        plt.legend()
    plt.savefig(save_name, bbox_inches="tight", dpi=180)
    plt.close()

def tsne16(input_data, input_target, save_name, show_legend=False):
    pass
    tsne = TSNE(n_jobs=8, n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(input_data, None)
    label = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16"]
    colors = ["tab:gray", 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'yellow', 'lime', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:cyan','tab:olive', 'magenta', 'darkred', 'coral', 'teal']
    plt.figure(figsize=(4, 4))
    for i, (l, c) in enumerate(zip(label, colors)):
        index = input_target == i
        plt.scatter(*tsne_results[index].transpose(), c=c, label=l, linewidth=0)
    if show_legend:
        plt.legend()
    plt.savefig(save_name, bbox_inches="tight", dpi=180)
    plt.close()
