import random
import os
import numpy as np
from DiffLib.TrainTools import BaseTrainer
from DiffLib.TrainTools import LatestModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import hparams

os.environ['MASTER_PORT'] = str(random.randint(15000, 30000))
random.seed(hparams['seed'])
np.random.seed(hparams['seed'])

work_dir = hparams['work_dir']

my_trainer = BaseTrainer(
    checkpoint_callback=LatestModelCheckpoint(filepath=work_dir,verbose=True, monitor='val_loss',mode='min',
                                              num_ckpt_keep=hparams['num_ckpt_keep'],save_best=hparams['save_best'],
                                              period=1 if hparams['save_ckpt'] else 100000),
    logger=TensorBoardLogger(save_dir=work_dir, name='lightning_logs', version='lastest' ),
    gradient_clip_val=hparams['clip_grad_norm'],
    val_check_interval=hparams['val_check_interval'],
    row_log_interval=hparams['log_interval'],
    max_updates=hparams['max_updates'],
    num_sanity_val_steps=hparams['num_sanity_val_steps'] if not hparams['validate'] else 10000,
    accumulate_grad_batches=hparams['accumulate_grad_batches'])

if not hparams['infer']:  # train
    my_trainer.TrainRun()
else:
    my_trainer.testing = True
    my_trainer.TrainRun()