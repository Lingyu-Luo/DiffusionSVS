import matplotlib

matplotlib.use('Agg')
import glob


from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.callbacks import ModelCheckpoint

import numpy as np
import torch.optim
import torch.utils.data
import copy
import logging
import os
import re
import sys
import types
import torch
import torch.distributed as dist
import tqdm
from torch.optim.optimizer import Optimizer
from DiffLib.net import DiffNet
from DiffLib.text_encoder import TokenTextEncoder
from DiffLib.nsf_hifigan import NsfHifiGAN
from DiffLib.diffusion import GaussianDiffusion
from DiffLib.base_dataset import BaseDataset
from DiffLib.indexed_datasets import IndexedDataset
from DiffLib.pitch_utils import norm_interp_f0
from DiffLib.pitch_utils import denorm_f0
from DiffLib.plot import spec_to_figure, dur_to_figure, f0_to_figure
from DiffLib.phoneme import load_phoneme_list
from config import hparams


vocoder = None


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt



class GradientAccumulationScheduler:
    def __init__(self, scheduling: dict):
        if scheduling == {}:  # empty dict error
            raise TypeError("Empty dict cannot be interpreted correct")

        for key in scheduling.keys():
            if not isinstance(key, int) or not isinstance(scheduling[key], int):
                raise TypeError("All epoches and accumulation factor must be integers")

        minimal_epoch = min(scheduling.keys())
        if minimal_epoch < 1:
            msg = f"Epochs indexing from 1, epoch {minimal_epoch} cannot be interpreted correct"
            raise IndexError(msg)
        elif minimal_epoch != 1:  # if user didnt define first epoch accumulation factor
            scheduling.update({1: 1})

        self.scheduling = scheduling
        self.epochs = sorted(scheduling.keys())

    def on_epoch_begin(self, epoch, trainer):
        epoch += 1  # indexing epochs from 1
        for i in reversed(range(len(self.epochs))):
            if epoch >= self.epochs[i]:
                trainer.accumulate_grad_batches = self.scheduling.get(self.epochs[i])
                break


class LatestModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0, num_ckpt_keep=5,
                 save_weights_only=False, mode='auto', period=1, prefix='model', save_best=True):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        os.makedirs(filepath, exist_ok=True)
        self.num_ckpt_keep = num_ckpt_keep
        self.save_best = save_best
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_check = 0
        self.prefix = prefix
        self.best_k_models = {}
        # {filename: monitor}
        self.kth_best_model = ''
        self.save_top_k = 1
        self.task = None
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
            self.mode = 'min'
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
            self.mode = 'max'
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
                self.mode = 'max'
            else:
                self.monitor_op = np.less
                self.best = np.Inf
                self.mode = 'min'
        if os.path.exists(f'{self.filepath}/best_valid.npy'):
            self.best = np.load(f'{self.filepath}/best_valid.npy')[0]

    def get_all_ckpts(self):
        return sorted(glob.glob(f'{self.filepath}/{self.prefix}_ckpt_steps_*.ckpt'),
                      key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))

    def on_epoch_end(self, epoch,global_step,logs=None):
        logs = logs or {}
        self.epochs_since_last_check += 1
        best_filepath = f'{self.filepath}/{self.prefix}_ckpt_best.pt'
        if self.epochs_since_last_check >= self.period:
            self.epochs_since_last_check = 0
            filepath = f'{self.filepath}/{self.prefix}_ckpt_steps_{global_step}.ckpt'
            if self.verbose > 0:
                logging.info(f'Epoch {epoch:05d}@{global_step}: saving model to {filepath}')
            self._save_model(filepath)
            for old_ckpt in self.get_all_ckpts()[self.num_ckpt_keep:]:
                # TODO: test filesystem calls
                os.remove(old_ckpt)
                # subprocess.check_call(f'del "{old_ckpt}"', shell=True)
                if self.verbose > 0:
                    logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
            current = logs.get(self.monitor)
            if current is not None and self.save_best:
                if self.monitor_op(current, self.best):
                    self.best = current
                    if self.verbose > 0:
                        logging.info(
                            f'Epoch {epoch:05d}@{self.task.global_step}: {self.monitor} reached'
                            f' {current:0.5f} (best {self.best:0.5f}), saving model to'
                            f' {best_filepath} as top 1')
                    self._save_model(best_filepath)
                    np.save(f'{self.filepath}/best_valid.npy', [self.best])


class BaseTrainer:
    def __init__(
            self,
            logger=True,
            checkpoint_callback=True,
            default_save_path=None,
            gradient_clip_val=0,
            process_position=0,
            gpus=-1,
            log_gpu_memory=None,
            show_progress_bar=True,
            track_grad_norm=-1,
            check_val_every_n_epoch=1,
            accumulate_grad_batches=1,
            max_updates=1000,
            min_epochs=1,
            val_check_interval=1.0,
            log_save_interval=100,
            row_log_interval=10,
            print_nan_grads=False,
            weights_summary='full',
            num_sanity_val_steps=5,
            resume_from_checkpoint=None,
    ):
        self.log_gpu_memory = log_gpu_memory
        self.gradient_clip_val = gradient_clip_val
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self.track_grad_norm = track_grad_norm
        self.on_gpu = True if (gpus and torch.cuda.is_available()) else False
        self.use_dp = False
        self.use_ddp = False
        self.process_position = process_position
        self.weights_summary = weights_summary
        self.max_updates = max_updates
        self.min_epochs = min_epochs
        self.num_sanity_val_steps = num_sanity_val_steps
        self.print_nan_grads = print_nan_grads
        self.resume_from_checkpoint = resume_from_checkpoint
        self.default_save_path = default_save_path

        #GPU Memory related
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences

        #training model related
        self.model = None
        self.testing = False
        self.disable_validation = False
        self.scheduler = None
        self.lr_schedulers = []
        self.optimizers = None
        self.loaded_optimizer_states_dict = {}
        self.mse_loss_fn = torch.nn.MSELoss()
        self.loss_and_lambda = {}

        # training process state
        self.total_batch_idx = 0
        self.running_loss = []
        self.avg_loss = 0
        self.batch_idx = 0
        self.tqdm_metrics = {}
        self.callback_metrics = {}
        self.num_val_batches = 0
        self.num_training_batches = 0
        self.num_test_batches = 0
        self.get_train_dataloader = None
        self.get_test_dataloaders = None
        self.get_val_dataloaders = None
        self.is_iterable_train_dataloader = False
        self.example_input_array = None
        self.global_step = 0
        self.current_epoch = 0
        self.total_batches = 0
        self.training_losses_meter = None

        # configure checkpoint callback
        self.checkpoint_callback = checkpoint_callback
        self.checkpoint_callback.save_function = self.save_checkpoint
        self.weights_save_path = self.checkpoint_callback.filepath

        # accumulated grads
        self.configure_accumulated_gradients(accumulate_grad_batches)

        # allow int, string and gpu list
        self.data_parallel_device_ids = [
            int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x != '']
        if len(self.data_parallel_device_ids) == 0:
            self.root_gpu = None
            self.on_gpu = False
        else:
            self.root_gpu = self.data_parallel_device_ids[0]
            self.on_gpu = True

        # distributed backend choice
        self.use_ddp = False
        self.use_dp = False
        self.single_gpu = False
        self.distributed_backend = 'ddp' if self.num_gpus > 0 else 'dp'
        self.set_distributed_mode(self.distributed_backend)

        self.proc_rank = 0
        self.world_size = 1
        self.node_rank = 0

        # can't init progress bar here because starting a new process
        # means the progress_bar won't survive pickling
        self.show_progress_bar = show_progress_bar

        # logging
        self.log_save_interval = log_save_interval
        self.val_check_interval = val_check_interval
        self.logger = logger
        self.logger.rank = 0
        self.row_log_interval = row_log_interval

        # 读入音素字典
        phone_list, pinyin2phs, phone_id, id_phoneme = load_phoneme_list(hparams['g2p_dictionary'])
        # 音素编码
        self.phone_encoder = TokenTextEncoder(vocab_list=phone_list, replace_oov=',')


    @property
    def num_gpus(self):
        gpus = self.data_parallel_device_ids
        if gpus is None:
            return 0
        else:
            return len(gpus)

    @property
    def data_parallel(self):
        return self.use_dp or self.use_ddp

    def get_model(self):
        is_dp_module = isinstance(self.model, (DDP, DP))
        model = self.model.module if is_dp_module else self.model
        return model

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    def TrainRun(self):
        mel_losses = hparams['mel_loss'].split("|")
        self.loss_and_lambda = {}
        for i, l in enumerate(mel_losses):
            if l == '':
                continue
            if ':' in l:
                l, lbd = l.split(":")
                lbd = float(lbd)
            else:
                lbd = 1.0
            self.loss_and_lambda[l] = lbd
        print("| Mel losses:", self.loss_and_lambda)
        self.sil_ph = self.phone_encoder.sil_phonemes()

        mel_bins = hparams['audio_num_mel_bins']
        global vocoder
        vocoder = NsfHifiGAN()
        vocoder.model.eval()
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=mel_bins, denoise_fn=DiffNet(mel_bins),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        print_arch(self.model)
        if not self.testing:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=hparams['lr'],
                betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
                weight_decay=hparams['weight_decay'])
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)
            self.optimizers, self.lr_schedulers = self.init_optimizers([optimizer])
        if self.single_gpu:
            self.model.cuda(self.root_gpu)
            vocoder.model.cuda(self.root_gpu)

        self.run_pretrain_routine(self.model)
        return 1

    def init_optimizers(self, optimizers):

        # single optimizer
        if isinstance(optimizers, Optimizer):
            return [optimizers], []

        # two lists
        elif len(optimizers) == 2 and isinstance(optimizers[0], list):
            optimizers, lr_schedulers = optimizers
            return optimizers, lr_schedulers

        # single list or tuple
        elif isinstance(optimizers, list) or isinstance(optimizers, tuple):
            return optimizers, []

    def training_step(self, sample, batch_idx, optimizer_idx=-1):
        loss_ret = self._training_step(sample, batch_idx, optimizer_idx)
        self.opt_idx = optimizer_idx
        if loss_ret is None:
            return {'loss': None}
        total_loss, log_outputs = loss_ret
        log_outputs = tensors_to_scalars(log_outputs)
        for k, v in log_outputs.items():
            if k not in self.training_losses_meter:
                self.training_losses_meter[k] = AvgrageMeter()
            if not np.isnan(v):
                self.training_losses_meter[k].update(v)
        self.training_losses_meter['total_loss'].update(total_loss.item())

        try:
            log_outputs['lr'] = self.scheduler.get_lr()
            if isinstance(log_outputs['lr'], list):
                log_outputs['lr'] = log_outputs['lr'][0]
        except:
            pass

        # log_outputs['all_loss'] = total_loss.item()
        progress_bar_log = log_outputs
        tb_log = {f'tr/{k}': v for k, v in log_outputs.items()}
        return {
            'loss': total_loss,
            'progress_bar': progress_bar_log,
            'log': tb_log
        }

    def _training_step(self, sample, batch_idx, _):
        log_outputs = self.run_model(self.model, sample)
        total_loss = sum([v for v in log_outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        log_outputs['batch_size'] = sample['txt_tokens'].size()[0]
        log_outputs['lr'] = self.scheduler.get_lr()[0]
        return total_loss, log_outputs

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

    def on_epoch_end(self):
        loss_outputs = {k: round(v.avg, 4) for k, v in self.training_losses_meter.items()}
        print(f"\n==============\n "
              f"Epoch {self.current_epoch} ended. Steps: {self.global_step}. {loss_outputs}"
              f"\n==============\n")

    def run_pretrain_routine(self, model):
        """Sanity check a few things before starting actual training.

        :param model:
        """
        # set local properties on the model
        #self.copy_trainer_model_properties(ref_model)

        # link up experiment object
        if self.logger is not None:
            #ref_model.logger = self.logger
            self.logger.save()

        if self.use_ddp:
            dist.barrier()

        # set up checkpoint callback
        # self.configure_checkpoint_callback()

        # transfer data loaders from model
        self.get_dataloaders(model)

        # restore training and model before hpc call
        #yaosiqi debug
        self.restore_weights(model)

        # when testing requested only run test and return
        if self.testing:
            self.run_evaluation(test=True)
            return

        # check if we should run validation during training
        self.disable_validation = self.num_val_batches == 0

        # run tiny validation (if validation defined)
        # to make sure program won't crash during val
        if not self.disable_validation and self.num_sanity_val_steps > 0:
            # init progress bars for validation sanity check
            pbar = tqdm.tqdm(desc='Validation sanity check',
                             total=self.num_sanity_val_steps * len(self.get_val_dataloaders),
                             leave=False, position=2 * self.process_position,
                             disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch')
            self.main_progress_bar = pbar
            # dummy validation progress bar
            self.val_progress_bar = tqdm.tqdm(disable=True)

            self.evaluate(model, self.get_val_dataloaders, self.num_sanity_val_steps, self.testing)

            # close progress bars
            self.main_progress_bar.close()
            self.val_progress_bar.close()

        # init progress bar
        pbar = tqdm.tqdm(leave=True, position=2 * self.process_position,
                         disable=not self.show_progress_bar, dynamic_ncols=True, unit='batch',
                         file=sys.stdout)
        self.main_progress_bar = pbar

        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()

        # CORE TRAINING LOOP
        self.train()

    def test(self, model):
        self.testing = True
        self.fit(model)

    @property
    def training_tqdm_dict(self):
        tqdm_dict = {
            'step': '{}'.format(self.global_step),
        }
        tqdm_dict.update(self.tqdm_metrics)
        return tqdm_dict

    # --------------------
    # restore ckpt
    # --------------------
    def restore_weights(self, model):
        """
        To restore weights we have two cases.
        First, attempt to restore hpc weights. If successful, don't restore
        other weights.

        Otherwise, try to restore actual weights
        :param model:
        :return:
        """
        # clear cache before restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        if self.resume_from_checkpoint is not None:
            self.restore(self.resume_from_checkpoint, on_gpu=self.on_gpu)
        else:
            # restore weights if same exp version
            self.restore_state_if_checkpoint_exists(model)

        # wait for all models to restore weights
        if self.use_ddp:
            # wait for all processes to catch up
            dist.barrier()

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

    def restore_state_if_checkpoint_exists(self, model):
        did_restore = False

        # do nothing if there's not dir or callback
        no_ckpt_callback = (self.checkpoint_callback is None) or (not self.checkpoint_callback)
        if no_ckpt_callback or not os.path.exists(self.checkpoint_callback.filepath):
            return did_restore

        # restore trainer state and model if there is a weight for this experiment
        last_steps = -1
        last_ckpt_name = None

        # find last epoch
        checkpoints = os.listdir(self.checkpoint_callback.filepath)
        for name in checkpoints:
            if '.ckpt' in name and not name.endswith('part'):
                if 'steps_' in name:
                    steps = name.split('steps_')[1]
                    steps = int(re.sub('[^0-9]', '', steps))

                    if steps > last_steps:
                        last_steps = steps
                        last_ckpt_name = name

        # restore last checkpoint
        if last_ckpt_name is not None:
            last_ckpt_path = os.path.join(self.checkpoint_callback.filepath, last_ckpt_name)
            self.restore(last_ckpt_path, self.on_gpu)
            logging.info(f'model and trainer restored from checkpoint: {last_ckpt_path}')
            print(f'model and trainer restored from checkpoint: {last_ckpt_path}')
            did_restore = True

        return did_restore

    def restore(self, checkpoint_path, on_gpu):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # load model state
        model = self.model
        # restore model data
        state_dict = checkpoint["state_dict"]
        state_dict = {k[len('model') + 1:]: v for k, v in state_dict.items()
                      if k.startswith(f'model.')}
        # load the state_dict on the model automatically
        model.load_state_dict(state_dict, strict=True)
        if on_gpu:
            model.cuda(self.root_gpu)
        # load training state (affects trainer only)
        self.restore_training_state(checkpoint)
        del checkpoint

        try:
            if dist.is_initialized() and dist.get_rank() > 0:
                return
        except Exception as e:
            print(e)
            return

    def restore_training_state(self, checkpoint):
        """
        Restore trainer state.
        Model will get its change to update
        :param checkpoint:
        :return:
        """
        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            self.checkpoint_callback.best = checkpoint['checkpoint_callback_best']

        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']

        if self.testing:
            return

        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            optimizer.load_state_dict(opt_state)

            # move optimizer to GPU 1 weight at a time
            # avoids OOM
            if self.root_gpu is not None:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.root_gpu)

        # restore the lr schedulers
        lr_schedulers = checkpoint['lr_schedulers']
        for scheduler, lrs_state in zip(self.lr_schedulers, lr_schedulers):
            scheduler.load_state_dict(lrs_state)

    # --------------------
    # MODEL SAVE CHECKPOINT
    # --------------------
    def _atomic_save(self, checkpoint, filepath):
        """Saves a checkpoint atomically, avoiding the creation of incomplete checkpoints.

        This will create a temporary checkpoint with a suffix of ``.part``, then copy it to the final location once
        saving is finished.

        Args:
            checkpoint (object): The object to save.
                Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
                accepts.
            filepath (str|pathlib.Path): The path to which the checkpoint will be saved.
                This points to the file that the checkpoint will be stored in.
        """
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)

    def save_checkpoint(self, filepath):
        checkpoint = self.dump_checkpoint()
        self._atomic_save(checkpoint, filepath)

    def dump_checkpoint(self):

        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step
        }

        if self.checkpoint_callback is not None and self.checkpoint_callback is not False:
            checkpoint['checkpoint_callback_best'] = self.checkpoint_callback.best

        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states

        # save lr schedulers
        lr_schedulers = []
        for i, scheduler in enumerate(self.lr_schedulers):
            lr_schedulers.append(scheduler.state_dict())

        checkpoint['lr_schedulers'] = lr_schedulers

        # add the hparams and state_dict from the model
        model = self.model
        state_dict = model.state_dict()
        state_dict = {'model.'+k: v for k, v in state_dict.items()}
        checkpoint['state_dict'] = state_dict

        return checkpoint

    def copy_trainer_model_properties(self, model):
        if isinstance(model, DP):
            ref_model = model.module
        elif isinstance(model, DDP):
            ref_model = model.module
        else:
            ref_model = model

        for m in [model, ref_model]:
            m.trainer = self
            m.on_gpu = self.on_gpu
            m.use_dp = self.use_dp
            m.use_ddp = self.use_ddp
            m.testing = self.testing
            m.single_gpu = self.single_gpu

    def transfer_batch_to_gpu(self, batch, gpu_id):
        # base case: object can be directly moved using `cuda` or `to`
        if callable(getattr(batch, 'cuda', None)):
            return batch.cuda(gpu_id, non_blocking=True)

        elif callable(getattr(batch, 'to', None)):
            return batch.to(torch.device('cuda', gpu_id), non_blocking=True)

        # when list
        elif isinstance(batch, list):
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return batch

        # when tuple
        elif isinstance(batch, tuple):
            batch = list(batch)
            for i, x in enumerate(batch):
                batch[i] = self.transfer_batch_to_gpu(x, gpu_id)
            return tuple(batch)

        # when dict
        elif isinstance(batch, dict):
            for k, v in batch.items():
                batch[k] = self.transfer_batch_to_gpu(v, gpu_id)

            return batch

        # nothing matches, return the value as is without transform
        return batch

    def set_distributed_mode(self, distributed_backend):
        # skip for CPU
        if self.num_gpus == 0:
            return

        # single GPU case
        # in single gpu case we allow ddp so we can train on multiple
        # nodes, 1 gpu per node
        elif self.num_gpus == 1:
            self.single_gpu = True
            self.use_dp = False
            self.use_ddp = False
            self.root_gpu = 0
            self.data_parallel_device_ids = [0]
        else:
            if distributed_backend is not None:
                self.use_dp = distributed_backend == 'dp'
                self.use_ddp = distributed_backend == 'ddp'
            elif distributed_backend is None:
                self.use_dp = True
                self.use_ddp = False

        logging.info(f'gpu available: {torch.cuda.is_available()}, used: {self.on_gpu}')

    def ddp_train(self, gpu_idx, model):
        """
        Entry point into a DP thread
        :param gpu_idx:
        :param model:
        :param cluster_obj:
        :return:
        """
        # otherwise default to node rank 0
        self.node_rank = 0

        # show progressbar only on progress_rank 0
        self.show_progress_bar = self.show_progress_bar and self.node_rank == 0 and gpu_idx == 0

        # determine which process we are and world size
        if self.use_ddp:
            self.proc_rank = self.node_rank * self.num_gpus + gpu_idx
            self.world_size = self.num_gpus

        # let the exp know the rank to avoid overwriting logs
        if self.logger is not None:
            self.logger.rank = self.proc_rank

        # set up server using proc 0's ip address
        # try to init for 20 times at max in case ports are taken
        # where to store ip_table
        model.trainer = self
        model.init_ddp_connection(self.proc_rank, self.world_size)

        # CHOOSE OPTIMIZER
        # allow for lr schedulers as well
        model.model = model.build_model()
        if not self.testing:
            self.optimizers, self.lr_schedulers = self.init_optimizers(model.configure_optimizers())

        # MODEL
        # copy model to each gpu
        if self.distributed_backend == 'ddp':
            torch.cuda.set_device(gpu_idx)
        model.cuda(gpu_idx)

        # set model properties before going into wrapper
        self.copy_trainer_model_properties(model)

        # override root GPU
        self.root_gpu = gpu_idx

        if self.distributed_backend == 'ddp':
            device_ids = [gpu_idx]
        else:
            device_ids = None

        # allow user to configure ddp
        model = model.configure_ddp(model, device_ids)

        # continue training routine
        self.run_pretrain_routine(model)

    def resolve_root_node_address(self, root_node):
        if '[' in root_node:
            name = root_node.split('[')[0]
            number = root_node.split(',')[0]
            if '-' in number:
                number = number.split('-')[0]

            number = re.sub('[^0-9]', '', number)
            root_node = name + number

        return root_node

    def log_metrics(self, metrics, grad_norm_dic, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        :param grad_norm_dic:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # add norms
        metrics.update(grad_norm_dic)

        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0 and self.logger is not None:
            self.logger.log_metrics(scalar_metrics, step=step)
            self.logger.save()

    def add_tqdm_metrics(self, metrics):
        for k, v in metrics.items():
            if type(v) is torch.Tensor:
                v = v.item()

            self.tqdm_metrics[k] = v

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics

    def process_output(self, output, train=False):
        """Reduces output according to the training mode.

        Separates loss from logging and tqdm metrics
        :param output:
        :return:
        """
        # ---------------
        # EXTRACT CALLBACK KEYS
        # ---------------
        # all keys not progress_bar or log are candidates for callbacks
        callback_metrics = {}
        for k, v in output.items():
            if k not in ['progress_bar', 'log', 'hiddens']:
                callback_metrics[k] = v

        if train and self.use_dp:
            num_gpus = self.num_gpus
            callback_metrics = self.reduce_distributed_output(callback_metrics, num_gpus)

        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()

        # ---------------
        # EXTRACT PROGRESS BAR KEYS
        # ---------------
        try:
            progress_output = output['progress_bar']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp:
                num_gpus = self.num_gpus
                progress_output = self.reduce_distributed_output(progress_output, num_gpus)

            progress_bar_metrics = progress_output
        except Exception:
            progress_bar_metrics = {}

        # ---------------
        # EXTRACT LOGGING KEYS
        # ---------------
        # extract metrics to log to experiment
        try:
            log_output = output['log']

            # reduce progress metrics for tqdm when using dp
            if train and self.use_dp:
                num_gpus = self.num_gpus
                log_output = self.reduce_distributed_output(log_output, num_gpus)

            log_metrics = log_output
        except Exception:
            log_metrics = {}

        # ---------------
        # EXTRACT LOSS
        # ---------------
        # if output dict doesn't have the keyword loss
        # then assume the output=loss if scalar
        loss = None
        if train:
            try:
                loss = output['loss']
            except Exception:
                if type(output) is torch.Tensor:
                    loss = output
                else:
                    raise RuntimeError(
                        'No `loss` value in the dictionary returned from `model.training_step()`.'
                    )

            # when using dp need to reduce the loss
            if self.use_dp:
                loss = self.reduce_distributed_output(loss, self.num_gpus)

        # ---------------
        # EXTRACT HIDDEN
        # ---------------
        hiddens = output.get('hiddens')

        # use every metric passed in as a candidate for callback
        callback_metrics.update(progress_bar_metrics)
        callback_metrics.update(log_metrics)

        # convert tensors to numpy
        for k, v in callback_metrics.items():
            if isinstance(v, torch.Tensor):
                callback_metrics[k] = v.item()

        return loss, progress_bar_metrics, log_metrics, callback_metrics, hiddens

    def reduce_distributed_output(self, output, num_gpus):
        if num_gpus <= 1:
            return output

        # when using DP, we get one output per gpu
        # average outputs and return
        if type(output) is torch.Tensor:
            return output.mean()

        for k, v in output.items():
            # recurse on nested dics
            if isinstance(output[k], dict):
                output[k] = self.reduce_distributed_output(output[k], num_gpus)

            # do nothing when there's a scalar
            elif isinstance(output[k], torch.Tensor) and output[k].dim() == 0:
                pass

            # reduce only metrics that have the same number of gpus
            elif output[k].size(0) == num_gpus:
                reduced = torch.mean(output[k])
                output[k] = reduced
        return output

    def clip_gradients(self):
        if self.gradient_clip_val > 0:
            model = self.model
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)

    def print_nan_gradients(self):
        model = self.get_model()
        for param in model.parameters():
            if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                logging.info(param, param.grad)

    def configure_accumulated_gradients(self, accumulate_grad_batches):
        self.accumulate_grad_batches = None

        if isinstance(accumulate_grad_batches, dict):
            self.accumulation_scheduler = GradientAccumulationScheduler(accumulate_grad_batches)
        elif isinstance(accumulate_grad_batches, int):
            schedule = {1: accumulate_grad_batches}
            self.accumulation_scheduler = GradientAccumulationScheduler(schedule)
        else:
            raise TypeError("Gradient accumulation supports only int and dict types")

    def get_dataloaders(self, model):
        if not self.testing:
            self.init_train_dataloader(model)
            self.init_val_dataloader(model)
        else:
            self.init_test_dataloader(model)

        if self.use_ddp:
            dist.barrier()
            if not self.testing:
                self.get_train_dataloader()
                self.get_val_dataloaders()
            else:
                self.get_test_dataloaders()

    def init_train_dataloader(self, model):
        self.fisrt_epoch = True
        self.get_train_dataloader = self.get_dataloader('train',shuffle=True)
        if isinstance(self.get_train_dataloader, torch.utils.data.DataLoader):
            self.num_training_batches = len(self.get_train_dataloader)
            self.num_training_batches = int(self.num_training_batches)
        else:
            self.num_training_batches = float('inf')
            self.is_iterable_train_dataloader = True
        if isinstance(self.val_check_interval, int):
            self.val_check_batch = self.val_check_interval
        else:
            self._percent_range_check('val_check_interval')
            self.val_check_batch = int(self.num_training_batches * self.val_check_interval)
            self.val_check_batch = max(1, self.val_check_batch)
            #Modified for debug
            self.val_check_batch = 2

    def get_dataloader(self, prefix,shuffle=False):
        _data_set = FastSpeechDataset(prefix, shuffle=shuffle)
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if prefix != 'train':
            self.max_tokens = 10000
            self.max_sentences = 1

        self.max_tokens *= devices_cnt
        self.max_sentences *= devices_cnt
        indices = _data_set.ordered_indices()
        batch_sampler = batch_by_size(
            indices, _data_set.num_tokens, max_tokens=self.max_tokens, max_sentences=self.max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )
        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler

        #num_workers = _data_set.num_workers
        num_workers = 0

        return torch.utils.data.DataLoader(_data_set,
                                           collate_fn=_data_set.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    def init_val_dataloader(self, model):
        self.get_val_dataloaders = [self.get_dataloader('valid',shuffle=False)]
        self.num_val_batches = 0
        if self.get_val_dataloaders is not None:
            if isinstance(self.get_val_dataloaders, torch.utils.data.DataLoader):
                self.num_val_batches = sum(len(dataloader) for dataloader in self.get_val_dataloaders)
                self.num_val_batches = int(self.num_val_batches)
            else:
                self.num_val_batches = float('inf')

    def init_test_dataloader(self, model):
        self.get_test_dataloaders = self.get_dataloader('test',shuffle=False)
        if self.get_test_dataloaders is not None:
            if isinstance(self.get_test_dataloaders, torch.utils.data.DataLoader):
                self.num_test_batches = sum(len(dataloader) for dataloader in self.get_test_dataloaders)
                self.num_test_batches = int(self.num_test_batches)
            else:
                self.num_test_batches = float('inf')

    def evaluate(self, model, dataloaders, max_batches, test=False):
        """Run evaluation code.

        :param model: PT model
        :param dataloaders: list of PT dataloaders
        :param max_batches: Scalar
        :param test: boolean
        :return:
        """
        # enable eval mode
        model.zero_grad()
        model.eval()

        # disable gradients to save memory
        torch.set_grad_enabled(False)

        # bookkeeping
        outputs = []

        # run evaluating
        for dataloader_idx, dataloader in enumerate(dataloaders):
            dl_outputs = []
            for batch_idx, batch in enumerate(dataloader):

                if batch is None:  # pragma: no cover
                    continue

                # stop short when on fast_dev_run (sets max_batch=1)
                if batch_idx >= max_batches:
                    break

                # -----------------
                # RUN EVALUATION STEP
                # -----------------
                output = self.evaluation_forward(model,
                                                 batch,
                                                 batch_idx,
                                                 dataloader_idx,
                                                 test)

                # track outputs for collation
                dl_outputs.append(output)

                # batch done
                if test:
                    self.test_progress_bar.update(1)
                else:
                    self.val_progress_bar.update(1)
            outputs.append(dl_outputs)

        # with a single dataloader don't pass an array
        if len(dataloaders) == 1:
            outputs = outputs[0]

        # give model a chance to do something with the outputs (and method defined)
        if test:
            eval_results_ = self.test_end(outputs)
        else:
            eval_results_ = self.validation_end(outputs)
        eval_results = eval_results_

        # enable gradients to save memory
        torch.set_grad_enabled(True)

        return eval_results

    def run_evaluation(self, test=False):
        # when testing make sure user defined a test step

        # select dataloaders
        if test:
            dataloaders = self.get_test_dataloaders
            max_batches = self.num_test_batches
        else:
            # val
            dataloaders = self.get_val_dataloaders
            max_batches = self.num_val_batches

        # init validation or test progress bar
        # main progress bar will already be closed when testing so initial position is free
        position = 2 * self.process_position + (not test)
        desc = 'Testing' if test else 'Validating'
        pbar = tqdm.tqdm(desc=desc, total=max_batches, leave=test, position=position,
                         disable=not self.show_progress_bar, dynamic_ncols=True,
                         unit='batch', file=sys.stdout)
        setattr(self, f'{"test" if test else "val"}_progress_bar', pbar)

        # run evaluation
        eval_results = self.evaluate(self.model,
                                     dataloaders,
                                     max_batches,
                                     test)
        if eval_results is not None:
            _, prog_bar_metrics, log_metrics, callback_metrics, _ = self.process_output(
                eval_results)

            # add metrics to prog bar
            self.add_tqdm_metrics(prog_bar_metrics)

            # log metrics
            self.log_metrics(log_metrics, {})

            # track metrics for callbacks
            self.callback_metrics.update(callback_metrics)


        # add model specific metrics
        tqdm_metrics = self.training_tqdm_dict
        if not test:
            self.main_progress_bar.set_postfix(**tqdm_metrics)

        # close progress bar
        if test:
            self.test_progress_bar.close()
        else:
            self.val_progress_bar.close()

        # model checkpointing
        if self.proc_rank == 0 and self.checkpoint_callback is not None and not test:
            self.checkpoint_callback.on_epoch_end(epoch=self.current_epoch,
                                                  global_step=self.global_step,
                                                  logs=self.callback_metrics)

    def evaluation_forward(self, model, batch, batch_idx, dataloader_idx, test=False):
        # make dataloader_idx arg in validation_step optional
        args = [batch, batch_idx]

        # handle DP, DDP forward
        if self.use_ddp or self.use_dp:
            output = model(*args)
            return output

        # single GPU
        if self.single_gpu:
            # for single GPU put inputs on gpu manually
            root_gpu = 0
            if isinstance(self.data_parallel_device_ids, list):
                root_gpu = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(batch, root_gpu)
            args[0] = batch

        # CPU
        if test:
            output = self.test_step(*args)
        else:
            output = self.validation_step(*args)

        return output


    def run_model(self, model, sample, return_output=False, infer=False):
        '''
            steps:
            1. run the full model, calc the main loss
            2. calculate loss for dur_predictor, pitch_predictor, energy_predictor
        '''
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph'] # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        energy = sample['energy']

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed,
                       ref_mels=target, f0=f0, uv=uv, energy=energy, infer=infer)

        losses = {}
        if 'diff_loss' in output:
            losses['mel'] = output['diff_loss']
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        txt_tokens = sample['txt_tokens']  # [B, T_t]

        target = sample['mels']  # [B, T_s, 80]
        energy = sample['energy']
        # fs2_mel = sample['fs2_mels']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        mel2ph = sample['mel2ph']
        f0 = sample['f0']

        outputs['losses'] = {}

        outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True, infer=False)

        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            model_out = self.model(
                txt_tokens, spk_embed=spk_embed, mel2ph=mel2ph, f0=f0, uv=None, energy=energy, ref_mels=None, infer=True,
                pitch_midi=sample.get('pitch_midi'), midi_dur=sample.get('midi_dur'), is_slur=sample.get('is_slur'))

            if hparams.get('pe_enable') is not None and hparams['pe_enable']:
                gt_f0 = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
                pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            else:
                gt_f0 = denorm_f0(sample['f0'], sample['uv'], hparams)
                pred_f0 = gt_f0
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'], name=f'diffmel_{batch_idx}')

        return outputs


    def validation_end(self, outputs):
        loss_output = self._validation_end(outputs)
        print(f"\n==============\n "
              f"valid results: {loss_output}"
              f"\n==============\n")
        return {
            'log': {f'val/{k}': v for k, v in loss_output.items()},
            'val_loss': loss_output['total_loss']
        }

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': AvgrageMeter(),
        }
        for output in outputs:
            n = output['nsamples']
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = AvgrageMeter()
                all_losses_meter[k].update(v, n)
            all_losses_meter['total_loss'].update(output['total_loss'], n)
        return {k: round(v.avg, 4) for k, v in all_losses_meter.items()}

    ############
    # validation plots
    ############
    def plot_wav(self, batch_idx, gt_wav, wav_out, is_mel=False, gt_f0=None, f0=None, name=None):
        gt_wav = gt_wav[0].cpu().numpy()
        wav_out = wav_out[0].cpu().numpy()
        gt_f0 = gt_f0[0].cpu().numpy()
        f0 = f0[0].cpu().numpy()
        if is_mel:
            gt_wav = vocoder.spec2wav(gt_wav, f0=gt_f0)
            wav_out = vocoder.spec2wav(wav_out, f0=f0)
        self.logger.experiment.add_audio(f'gt_{batch_idx}', gt_wav, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)
        self.logger.experiment.add_audio(f'wav_{batch_idx}', wav_out, sample_rate=hparams['audio_sample_rate'],
                                         global_step=self.global_step)

    ############
    # validation plots
    ############
    def plot_mel(self, batch_idx, spec, spec_out, name=None):
        spec_cat = torch.cat([spec, spec_out], -1)
        name = f'mel_{batch_idx}' if name is None else name
        vmin = hparams['mel_vmin']
        vmax = hparams['mel_vmax']
        self.logger.experiment.add_figure(name, spec_to_figure(spec_cat[0], vmin, vmax), self.global_step)

    def plot_dur(self, batch_idx, sample, model_out):
        T_txt = sample['txt_tokens'].shape[1]
        dur_gt = mel2ph_to_dur(sample['mel2ph'], T_txt)[0]
        dur_pred = self.model.dur_predictor.out2dur(model_out['dur']).float()
        txt = self.phone_encoder.decode(sample['txt_tokens'][0].cpu().numpy())
        txt = txt.split(" ")
        self.logger.experiment.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt), self.global_step)

    def plot_pitch(self, batch_idx, sample, model_out):
        f0 = sample['f0']
        if hparams['pitch_type'] == 'ph':
            mel2ph = sample['mel2ph']
            f0 = self.expand_f0_ph(f0, mel2ph)
            f0_pred = self.expand_f0_ph(model_out['pitch_pred'][:, :, 0], mel2ph)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, f0_pred[0]), self.global_step)
            return
        f0 = denorm_f0(f0, sample['uv'], hparams)
        if hparams['pitch_type'] == 'cwt':
            # cwt
            cwt_out = model_out['cwt']
            cwt_spec = cwt_out[:, :, :10]
            cwt = torch.cat([cwt_spec, sample['cwt_spec']], -1)
            self.logger.experiment.add_figure(f'cwt_{batch_idx}', spec_to_figure(cwt[0]), self.global_step)
            # f0
            f0_pred = cwt2f0(cwt_spec, model_out['f0_mean'], model_out['f0_std'], hparams['cwt_scales'])
            if hparams['use_uv']:
                assert cwt_out.shape[-1] == 11
                uv_pred = cwt_out[:, :, -1] > 0
                f0_pred[uv_pred > 0] = 0
            f0_cwt = denorm_f0(sample['f0_cwt'], sample['uv'], hparams)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], f0_cwt[0], f0_pred[0]), self.global_step)
        elif hparams['pitch_type'] == 'frame':
            # f0
            uv_pred = model_out['pitch_pred'][:, :, 1] > 0
            pitch_pred = denorm_f0(model_out['pitch_pred'][:, :, 0], uv_pred, hparams)
            self.logger.experiment.add_figure(
                f'f0_{batch_idx}', f0_to_figure(f0[0], None, pitch_pred[0]), self.global_step)


    def train(self):
        # run all epochs
        for epoch in range(self.current_epoch, 20): #for debug yaosiqi, modified 1000000->20
            # set seed for distributed sampler (enables shuffling for each epoch)
            if self.use_ddp and hasattr(self.get_train_dataloader.sampler, 'set_epoch'):
                self.get_train_dataloader.sampler.set_epoch(epoch)

            # update training progress in trainer and model
            self.current_epoch = epoch

            total_val_batches = 0
            if not self.disable_validation:
                # val can be checked multiple times in epoch
                is_val_epoch = (self.current_epoch + 1) % self.check_val_every_n_epoch == 0
                val_checks_per_epoch = self.num_training_batches // self.val_check_batch
                val_checks_per_epoch = val_checks_per_epoch if is_val_epoch else 0
                total_val_batches = self.num_val_batches * val_checks_per_epoch

            # total batches includes multiple val checks
            self.total_batches = self.num_training_batches + total_val_batches
            self.batch_loss_value = 0  # accumulated grads

            if self.is_iterable_train_dataloader:
                # for iterable train loader, the progress bar never ends
                num_iterations = None
            else:
                num_iterations = self.total_batches

            # reset progress bar
            # .reset() doesn't work on disabled progress bar so we should check
            desc = f'Epoch {epoch + 1}' if not self.is_iterable_train_dataloader else ''
            self.main_progress_bar.set_description(desc)

            # changing gradient according accumulation_scheduler
            self.accumulation_scheduler.on_epoch_begin(epoch, self)

            # -----------------
            # RUN TNG EPOCH
            # -----------------
            self.run_training_epoch()

            # update LR schedulers
            if self.lr_schedulers is not None:
                for lr_scheduler in self.lr_schedulers:
                    lr_scheduler.step(epoch=self.current_epoch)

        self.main_progress_bar.close()

        model.on_train_end()

        if self.logger is not None:
            self.logger.finalize("success")

    def run_training_epoch(self):
        # before epoch hook
        self.training_losses_meter = {'total_loss': AvgrageMeter()}

        # run epoch
        for batch_idx, batch in enumerate(self.get_train_dataloader):
            # stop epoch if we limited the number of training batches
            if batch_idx >= self.num_training_batches:
                break

            self.batch_idx = batch_idx

            model = self.model

            # ---------------
            # RUN TRAIN STEP
            # ---------------
            output = self.run_training_batch(batch, batch_idx)
            batch_result, grad_norm_dic, batch_step_metrics = output

            # when returning -1 from train_step, we end epoch early
            early_stop_epoch = batch_result == -1

            # ---------------
            # RUN VAL STEP
            # ---------------
            should_check_val = (
                    not self.disable_validation and self.global_step % self.val_check_batch == 0 and not self.fisrt_epoch)
            self.fisrt_epoch = False

            if should_check_val:
                self.run_evaluation(test=self.testing)

            # when logs should be saved
            should_save_log = (batch_idx + 1) % self.log_save_interval == 0 or early_stop_epoch
            if should_save_log:
                if self.proc_rank == 0 and self.logger is not None:
                    self.logger.save()

            # when metrics should be logged
            should_log_metrics = batch_idx % self.row_log_interval == 0 or early_stop_epoch
            if should_log_metrics:
                # logs user requested information to logger
                self.log_metrics(batch_step_metrics, grad_norm_dic)

            self.global_step += 1
            self.total_batch_idx += 1

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if early_stop_epoch:
                break
            if self.global_step > self.max_updates:
                print("| Training end..")
                exit()

        # epoch end hook
        self.on_epoch_end()


    def run_training_batch(self, batch, batch_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        all_callback_metrics = []

        # track metrics to log
        all_log_metrics = []

        if batch is None:
            return 0, grad_norm_dic, {}

        splits = [batch]
        self.hiddens = None
        for split_idx, split_batch in enumerate(splits):
            self.split_idx = split_idx

            # call training_step once per optimizer
            for opt_idx, optimizer in enumerate(self.optimizers):
                if optimizer is None:
                    continue
                # make sure only the gradients of the current optimizer's paramaters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if len(self.optimizers) > 1:
                    for param in self.get_model().parameters():
                        param.requires_grad = False
                    for group in optimizer.param_groups:
                        for param in group['params']:
                            param.requires_grad = True

                # wrap the forward step in a closure so second order methods work
                def optimizer_closure():
                    # forward pass
                    output = self.training_forward(
                        split_batch, batch_idx, opt_idx, self.hiddens)

                    closure_loss = output[0]
                    progress_bar_metrics = output[1]
                    log_metrics = output[2]
                    callback_metrics = output[3]
                    self.hiddens = output[4]
                    if closure_loss is None:
                        return None

                    # accumulate loss
                    # (if accumulate_grad_batches = 1 no effect)
                    closure_loss = closure_loss / self.accumulate_grad_batches

                    # backward pass
                    if closure_loss.requires_grad:
                        closure_loss.backward()

                    # track metrics for callbacks
                    all_callback_metrics.append(callback_metrics)

                    # track progress bar metrics
                    self.add_tqdm_metrics(progress_bar_metrics)
                    all_log_metrics.append(log_metrics)

                    return closure_loss

                # calculate loss
                loss = optimizer_closure()
                if loss is None:
                    continue

                # nan grads
                if self.print_nan_grads:
                    self.print_nan_gradients()

                # track total loss for logging (avoid mem leaks)
                self.batch_loss_value += loss.item()

                # gradient update with accumulated gradients
                if (self.batch_idx + 1) % self.accumulate_grad_batches == 0:

                    # track gradient norms when requested
                    if batch_idx % self.row_log_interval == 0:
                        if self.track_grad_norm > 0:
                            model = self.get_model()
                            grad_norm_dic = model.grad_norm(
                                self.track_grad_norm)

                    # clip gradients
                    self.clip_gradients()

                    # calls .step(), .zero_grad()
                    # override function to modify this behavior
                    self.optimizer_step(self.current_epoch, batch_idx, optimizer, opt_idx)

                    # calculate running loss for display
                    self.running_loss.append(self.batch_loss_value)
                    self.batch_loss_value = 0
                    self.avg_loss = np.mean(self.running_loss[-100:])

        # update progress bar
        self.main_progress_bar.update(1)
        self.main_progress_bar.set_postfix(**self.training_tqdm_dict)

        # collapse all metrics into one dict
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.callback_metrics.update({k: v for d in all_callback_metrics for k, v in d.items()})

        return 0, grad_norm_dic, all_log_metrics

    def training_forward(self, batch, batch_idx, opt_idx, hiddens):
        """
        Handle forward for each training case (distributed, single gpu, etc...)
        :param batch:
        :param batch_idx:
        :return:
        """
        # ---------------
        # FORWARD
        # ---------------
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx, opt_idx]

        # distributed forward
        if self.use_ddp or self.use_dp:
            output = self.model(*args)
        # single GPU forward
        elif self.single_gpu:
            gpu_id = 0
            if isinstance(self.data_parallel_device_ids, list):
                gpu_id = self.data_parallel_device_ids[0]
            batch = self.transfer_batch_to_gpu(copy.copy(batch), gpu_id)
            args[0] = batch
            output = self.training_step(*args)
        # CPU forward
        else:
            output = self.training_step(*args)

        # format and reduce outputs accordingly
        output = self.process_output(output, train=True)

        return output

    # ---------------
    # Utils
    # ---------------
    def is_function_implemented(self, f_name):
        model = self.get_model()
        f_op = getattr(model, f_name, None)
        return callable(f_op)

    def _percent_range_check(self, name):
        value = getattr(self, name)
        msg = f"`{name}` must lie in the range [0.0, 1.0], but got {value:.3f}."
        if name == "val_check_interval":
            msg += " If you want to disable validation set `val_percent_check` to 0.0 instead."

        if not 0. <= value <= 1.:
            raise ValueError(msg)

def print_arch(model, model_name='model'):
    print(f"| {model_name} Arch: ", model)
    num_params(model, model_name=model_name)


def num_params(model, print_out=True, model_name="model"):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print(f'| {model_name} Trainable Parameters: %.3fM' % parameters)
    return parameters



class FastSpeechDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None
        # self.name2spk_id={}

        # pitch stats
        f0_stats_fn = f'{self.data_dir}/train_f0s_mean_std.npy'
        if os.path.exists(f0_stats_fn):
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = np.load(f0_stats_fn)
            hparams['f0_mean'] = float(hparams['f0_mean'])
            hparams['f0_std'] = float(hparams['f0_std'])
        else:
            hparams['f0_mean'], hparams['f0_std'] = self.f0_mean, self.f0_std = None, None

        if prefix == 'test':
            if hparams['test_input_dir'] != '':
                self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            else:
                if hparams['num_test_samples'] > 0:
                    self.avail_idxs = list(range(hparams['num_test_samples'])) + hparams['test_ids']
                    self.sizes = [self.sizes[i] for i in self.avail_idxs]

        if hparams['pitch_type'] == 'cwt':
            _, hparams['cwt_scales'] = get_lf0_cwt(np.ones(10))

    def _get_item(self, index):
        if hasattr(self, 'avail_idxs') and self.avail_idxs is not None:
            index = self.avail_idxs[index]
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        energy = (spec.exp() ** 2).sum(-1).sqrt()
        mel2ph = torch.LongTensor(item['mel2ph'])[:max_frames] if 'mel2ph' in item else None
        f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        pitch = torch.LongTensor(item.get("pitch"))[:max_frames]
        # print(item.keys(), item['mel'].shape, spec.shape)
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "pitch": pitch,
            "energy": energy,
            "f0": f0,
            "uv": uv,
            "mel2ph": mel2ph,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if self.hparams['use_spk_embed']:
            sample["spk_embed"] = torch.Tensor(item['spk_embed'])
        if self.hparams['use_spk_id']:
            sample["spk_id"] = item['spk_id']
            # sample['spk_id'] = 0
            # for key in self.name2spk_id.keys():
            #     if key in item['item_name']:
            #         sample['spk_id'] = self.name2spk_id[key]
            #         break
        if self.hparams['pitch_type'] == 'cwt':
            cwt_spec = torch.Tensor(item['cwt_spec'])[:max_frames]
            f0_mean = item.get('f0_mean', item.get('cwt_mean'))
            f0_std = item.get('f0_std', item.get('cwt_std'))
            sample.update({"cwt_spec": cwt_spec, "f0_mean": f0_mean, "f0_std": f0_std})
        elif self.hparams['pitch_type'] == 'ph':
            f0_phlevel_sum = torch.zeros_like(phone).float().scatter_add(0, mel2ph - 1, f0)
            f0_phlevel_num = torch.zeros_like(phone).float().scatter_add(
                0, mel2ph - 1, torch.ones_like(f0)).clamp_min(1)
            sample["f0_ph"] = f0_phlevel_sum / f0_phlevel_num
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = collate_1d([s['txt_token'] for s in samples], 0)
        f0 = collate_1d([s['f0'] for s in samples], 0.0)
        pitch = collate_1d([s['pitch'] for s in samples])
        uv = collate_1d([s['uv'] for s in samples])
        energy = collate_1d([s['energy'] for s in samples], 0.0)
        mel2ph = collate_1d([s['mel2ph'] for s in samples], 0.0) \
            if samples[0]['mel2ph'] is not None else None
        mels = collate_2d([s['mel'] for s in samples], 0.0)
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel2ph': mel2ph,
            'energy': energy,
            'pitch': pitch,
            'f0': f0,
            'uv': uv,
        }

        if self.hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if self.hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids
        if self.hparams['pitch_type'] == 'cwt':
            cwt_spec = collate_2d([s['cwt_spec'] for s in samples])
            f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
            f0_std = torch.Tensor([s['f0_std'] for s in samples])
            batch.update({'cwt_spec': cwt_spec, 'f0_mean': f0_mean, 'f0_std': f0_std})
        elif self.hparams['pitch_type'] == 'ph':
            batch['f0'] = collate_1d([s['f0_ph'] for s in samples])

        return batch

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'basics.base_binarizer.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = os.path.basename(wav_fn)
            ph = txt = tg_fn = ''
            wav_fn = wav_fn
            encoder = None
            item = binarizer_cls.process_item(item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes




def batch_by_size(
        indices, num_tokens_fn, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1, distributed=False
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)
        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches

def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0

def collate_1d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None, shift_id=1):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
            dst[0] = shift_id
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def collate_2d(values, pad_idx=0, left_pad=False, shift_right=False, max_len=None):
    """Convert a list of 2d tensors into a padded 3d tensor."""
    size = max(v.size(0) for v in values) if max_len is None else max_len
    res = values[0].new(len(values), size, values[0].shape[1]).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if shift_right:
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def tensors_to_scalars(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        if type(v) is dict:
            v = tensors_to_scalars(v)
        new_metrics[k] = v
    return new_metrics