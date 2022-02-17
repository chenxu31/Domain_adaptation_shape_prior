from pathlib import Path
from typing import Union, Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter

from loss.IIDSegmentations import compute_joint_distribution
from loss.entropy import SimplexCrossEntropyLoss
from meters import Storage
from meters.SummaryWriter import SummaryWriter
from scheduler.customized_scheduler import RampScheduler
from utils import tqdm
from utils.general import path2Path, class2one_hot
from utils.image_save_utils import plot_joint_matrix
from utils.utils import set_environment, write_yaml, meters_register


class SourcebaselineTrainer:
    PROJECT_PATH = str(Path(__file__).parents[1])

    RUN_PATH = str(Path(PROJECT_PATH) / "runs")
    ARCHIVE_PATH = str(Path(PROJECT_PATH) / "archives")
    wholemeter_filename = "wholeMeter.csv"
    checkpoint_identifier = "last.pth"

    def __init__(
            self,
            model: nn.Module,
            optimizer,
            scheduler,
            TrainS_loader: Union[DataLoader, _BaseDataLoaderIter],
            TrainT_loader: Union[DataLoader, _BaseDataLoaderIter],
            valS_loader: Union[DataLoader, _BaseDataLoaderIter],
            valT_loader: Union[DataLoader, _BaseDataLoaderIter],
            weight_scheduler: RampScheduler,
            weight_cluster: RampScheduler,
            switch_bn,
            max_epoch: int = 100,
            save_dir: str = "base",
            checkpoint_path: str = None,
            device='cpu',
            config: dict = None,
            num_batches=200,
            *args,
            **kwargs
    ) -> None:
        self._save_dir: Path = Path(self.RUN_PATH) / str(save_dir)
        self._save_dir.mkdir(exist_ok=True, parents=True)
        self._start_epoch = 0
        if config:
            self._config = config.copy()
            self._config.pop("Config", None)
            write_yaml(self._config, save_dir=self._save_dir, save_name="config.yaml")
            set_environment(config.get("Environment"))

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._trainS_loader = TrainS_loader
        self._trainT_loader = TrainT_loader
        self._valS_loader = valS_loader
        self._valT_loader = valT_loader
        self._max_epoch = max_epoch
        self._num_batches = num_batches
        self._weight_scheduler = weight_scheduler
        self._weight_cluster = weight_cluster
        self.switch_bn = switch_bn
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.crossentropy = SimplexCrossEntropyLoss()
        self._storage = Storage(self._save_dir)
        self.writer = SummaryWriter(str(self._save_dir))
        c = self._config['Data_input']['num_class']
        self.meters = meters_register(c)

    def to(self, device):
        self.model.to(device=device)

    def run_step(self, s_data, t_data, cur_batch: int):
        S_img, S_target, S_filename = (
            s_data[0][0].to(self.device),
            s_data[0][1].to(self.device),
            s_data[1],
        )
        with self.switch_bn(self.model, 0):
            pred_S = self.model(S_img).softmax(1)
        onehot_targetS = class2one_hot(
            S_target.squeeze(1), self._config['Data_input']['num_class']
        )
        s_loss = self.crossentropy(pred_S, onehot_targetS)

        self.meters[f"train_dice"].add(
            pred_S.max(1)[1],
            S_target.squeeze(1),
            group_name=["_".join(x.split("_")[:-1]) for x in S_filename],
        )

        align_loss = torch.tensor(0, dtype=torch.float, device=pred_S.device)
        cluster_loss = torch.tensor(0, dtype=torch.float, device=pred_S.device)

        p_joint_S = compute_joint_distribution(
            x_out=pred_S,
            displacement_map=(self._config['DA']['displacement']['map_x'],
                              self._config['DA']['displacement']['map_y']))
        if cur_batch == 0:
            source_joint_fig = plot_joint_matrix(p_joint_S)
            self.writer.add_figure(tag=f"source_joint", figure=source_joint_fig, global_step=self.cur_epoch,
                                   close=True, )
        return s_loss, cluster_loss, align_loss

    def train_loop(
            self,
            trainS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            trainT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ):
        self.model.train()
        batch_indicator = tqdm(range(self._num_batches))
        batch_indicator.set_description(f"Training Epoch {epoch:03d}")
        report_dict, p_joint_S, p_joint_T = None, None, None

        for cur_batch, (batch_id, s_data, t_data) in enumerate(zip(batch_indicator, trainS_loader, trainT_loader)):
            self.optimizer.zero_grad()

            s_loss, cluster_loss, align_loss = self.run_step(s_data=s_data, t_data=t_data, cur_batch=cur_batch)
            loss = s_loss + self._weight_cluster.value * cluster_loss + self._weight_scheduler.value * align_loss
            # if using dual Mean Teacher
            # run_step
            # gradient backforward

            loss.backward()
            self.optimizer.step()

            self.meters['total_loss'].add(loss.item())
            self.meters['s_loss'].add(s_loss.item())
            self.meters['align_loss'].add(align_loss.item())
            self.meters['cluster_loss'].add(cluster_loss.item())

            report_dict = self.meters.statistics()
            batch_indicator.set_postfix_statics(report_dict, cache_time=20)
        batch_indicator.close()

        assert report_dict is not None
        return dict(report_dict)

    def eval_loop(
            self,
            valS_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            valT_loader: Union[DataLoader, _BaseDataLoaderIter] = None,
            epoch: int = 0,
            *args,
            **kwargs,
    ) -> Tuple[Any, Any]:
        self.model.eval()
        valS_indicator = tqdm(valS_loader)
        valS_indicator.set_description(f"ValS_Epoch {epoch:03d}")
        valT_indicator = tqdm(valT_loader)
        valT_indicator.set_description(f"ValT_Epoch {epoch:03d}")
        report_dict = {}
        for batch_idS, data_S in enumerate(valS_indicator):
            imageS, targetS, filenameS = (
                data_S[0][0].to(self.device),
                data_S[0][1].to(self.device),
                data_S[1]
            )
            with self.switch_bn(self.model, 0):
                preds_S = self.model(imageS).softmax(1)
            self.meters[f"valS_dice"].add(
                preds_S.max(1)[1],
                targetS.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filenameS])

            report_dict = self.meters.statistics()
            valS_indicator.set_postfix_statics(report_dict, cache_time=20)
        valS_indicator.close()

        for batch_idT, data_T in enumerate(valT_indicator):
            imageT, targetT, filenameT = (
                data_T[0][0].to(self.device),
                data_T[0][1].to(self.device),
                data_T[1]
            )
            with self.switch_bn(self.model, 1):
                preds_T = self.model(imageT).softmax(1)
            self.meters[f"valT_dice"].add(
                preds_T.max(1)[1],
                targetT.squeeze(1),
                group_name=["_".join(x.split("_")[:-1]) for x in filenameT])

            report_dict = self.meters.statistics()
            valT_indicator.set_postfix_statics(report_dict, cache_time=20)

        valT_indicator.close()
        assert report_dict is not None
        return dict(report_dict), self.meters["valT_dice"].summary()["DSC_mean"]

    def schedulerStep(self):
        self._weight_scheduler.step()
        self._weight_cluster.step()
        self.scheduler.step()

    def start_training(self):
        self.to(self.device)
        self.cur_epoch = 0

        for self.cur_epoch in range(self._start_epoch, self._max_epoch):
            self.meters.reset()
            self.meters['lr'].add(self.optimizer.param_groups.__getitem__(0).get('lr'))
            self.meters["weight"].add(self._weight_scheduler.value)

            train_metrics = self.train_loop(
                trainS_loader=self._trainS_loader,
                trainT_loader=self._trainT_loader,
                epoch=self.cur_epoch
            )

            with torch.no_grad():
                val_metric, _ = self.eval_loop(self._valS_loader, self._valT_loader, self.cur_epoch)

            with self._storage:
                self._storage.add_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)
                self.writer.add_scalars_from_meter_interface(tra=train_metrics, val=val_metric, epoch=self.cur_epoch)

            self.schedulerStep()
            self.save_checkpoint(self.state_dict(), self.cur_epoch)

    def inference(self, identifier="best.pth", *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        if self._checkpoint is None:
            self._checkpoint = self._save_dir
        assert Path(self._checkpoint).exists(), Path(self._checkpoint)
        assert (Path(self._checkpoint).is_dir() and identifier is not None) or (
                Path(self._checkpoint).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self._checkpoint) / identifier)
            if identifier is not None
            else self._checkpoint,
            map_location=torch.device("cpu"),
        )
        self.load_checkpoint(state_dict)
        self.model.to(self.device)
        # to be added
        # probably call self._eval() method.

    def state_dict(self) -> Dict[str, Any]:
        """
        return trainer's state dict. The dict is built by considering all the submodules having `state_dict` method.
        """
        state_dictionary = {}
        for module_name, module in self.__dict__.items():
            if hasattr(module, "state_dict"):
                state_dictionary[module_name] = module.state_dict()
        return state_dictionary

    def save_checkpoint(
            self, state_dict, current_epoch, save_dir=None, save_name=None
    ):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :return:
        """
        # save_best: bool = True if float(cur_score) > float(self._best_score) else False
        # if save_best:
        #     self._best_score = float(cur_score)
        state_dict["epoch"] = current_epoch
        # state_dict["best_score"] = float(self._best_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            # if save_best:
            #     torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def _load_state_dict(self, state_dict) -> None:
        """
        Load state_dict for submodules having "load_state_dict" method.
        :param state_dict:
        :return:
        """
        for module_name, module in self.__dict__.items():
            if hasattr(module, "load_state_dict"):
                try:
                    module.load_state_dict(state_dict[module_name])
                except KeyError as e:
                    print(f"Loading checkpoint error for {module_name}, {e}.")
                except RuntimeError as e:
                    print(f"Interface changed error for {module_name}, {e}")

    def load_checkpoint(self, state_dict) -> None:
        """
        load checkpoint to models, meters, best score and _start_epoch
        Can be extended by add more state_dict
        :param state_dict:
        :return:
        """
        self._load_state_dict(state_dict)
        self._best_score = state_dict["best_score"]
        self._start_epoch = state_dict["epoch"] + 1

    def load_checkpoint_from_path(self, checkpoint_path):
        checkpoint_path = path2Path(checkpoint_path)
        assert checkpoint_path.exists(), checkpoint_path
        if checkpoint_path.is_dir():
            state_dict = torch.load(
                str(Path(checkpoint_path) / self.checkpoint_identifier),
                map_location=torch.device("cpu"),
            )
        else:
            assert checkpoint_path.suffix == ".pth", checkpoint_path
            state_dict = torch.load(
                str(checkpoint_path), map_location=torch.device("cpu"),
            )
        self.load_checkpoint(state_dict)

    def clean_up(self, wait_time=3):
        """
        Do not touch
        :return:
        """
        import shutil
        import time

        time.sleep(wait_time)  # to prevent that the call_draw function is not ended.
        Path(self.ARCHIVE_PATH).mkdir(exist_ok=True, parents=True)
        sub_dir = self._save_dir.relative_to(Path(self.RUN_PATH))
        save_dir = Path(self.ARCHIVE_PATH) / str(sub_dir)
        if Path(save_dir).exists():
            shutil.rmtree(save_dir, ignore_errors=True)
        shutil.move(str(self._save_dir), str(save_dir))
        shutil.rmtree(str(self._save_dir), ignore_errors=True)
