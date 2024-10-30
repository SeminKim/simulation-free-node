import os
import torch
from data.load_data import Data
from models.base import BaseModel
from lightning.pytorch.cli import LightningCLI
from utils import *

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


class CustomLightningCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def before_fit(self):
        self.trainer.logger.log_hyperparams(self.config.fit.as_dict())

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--name", type=str, required=True)  # experiment name should be provided
        parser.link_arguments("trainer.max_steps", "model.init_args.total_steps")
        # automatic linking of arguments
        parser.link_arguments("name", "trainer.logger.init_args.name")
        parser.link_arguments("name", "trainer.default_root_dir", compute_fn=lambda x: os.path.join("logs", x))
        parser.link_arguments("data.task_type", "model.init_args.task_criterion", compute_fn=compute_task_criterion)
        parser.link_arguments("data.task_type", "model.init_args.metric_type", compute_fn=compute_metric_type)

    def before_instantiate_classes(self):
        mode = getattr(self.config, 'subcommand', None)
        if mode is None:
            return  # not running subcommand

        self.config[mode]['trainer']['gradient_clip_val'] = 1.0 if self.config[mode]['model'][
            'init_args']['method'] == 'node' or self.config[mode]['data']['dataset'] == 'uci' else 0.

        # set hidden_dim = latent_dim for uci dataset
        if self.config[mode]['data']['task_type'] == 'regression':
            assert self.config[mode]['data']['dataset'] == 'uci'
        if self.config[mode]['data']['dataset'] == 'uci':
            self.config[mode]['model']['init_args']['hidden_dim'] = self.config[mode]['model']['init_args']['latent_dim']

        # set save_dir of logger
        name = self.config[mode]['trainer']['logger']['init_args']['name']
        self.config[mode]['trainer']['logger']['init_args']['save_dir'] = os.path.join(
            self.config[mode]['trainer']['logger']['init_args']['save_dir'], name)
        os.makedirs(self.config[mode]['trainer']['logger']['init_args']['save_dir'], exist_ok=True)

    def instantiate_classes(self):
        '''
        Hacks to set the data_dim, output_dim and label scaler for UCI dataset.
        Overrides the instantiate_classes method in LightningCLI.
        '''
        mode = getattr(self.config, 'subcommand', None)
        if mode is None:  # not running subcommand
            assert self.config['data']['task_type'] != 'regression', 'debug code not implemented for uci'
            return super().instantiate_classes()

        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init, "data")

        if self.config[mode]['data']['dataset'] == 'uci':
            self.config[mode]['model']['init_args']['data_dim'] = self.datamodule.train_dataset.train_dim_x
            self.config[mode]['model']['init_args']['output_dim'] = self.datamodule.train_dataset.train_dim_y
            self.config_init = self.parser.instantiate_classes(self.config)
        self.model = self._get(self.config_init, "model")
        self._add_configure_optimizers_method_to_model(self.subcommand)
        self.trainer = self.instantiate_trainer()
        if self.config[mode]['model']['init_args']['label_scaler'] is True:
            self.datamodule.normalize_y = True
            self.model.label_scaler = self.datamodule.train_dataset.scaler_y

        # print total batch size
        per_gpu = self.config[mode]['data']['batch_size']
        num_gpus = self.trainer.world_size
        total = per_gpu * num_gpus
        print(f"Using total batch size {total} = {num_gpus} x {per_gpu}")


if __name__ == '__main__':
    cli = CustomLightningCLI(model_class=BaseModel,
                             subclass_mode_model=True,
                             datamodule_class=Data,
                             save_config_kwargs={"overwrite": True},
                             run=True,)
    print(f'Done.')
