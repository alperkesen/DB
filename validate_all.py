import os

import torch
from tqdm import tqdm

from experiment import Experiment
from data.data_loader import DistributedSampler


class Trainer:
    def __init__(self, experiment: Experiment):
        self.init_device()

        self.experiment = experiment
        self.structure = experiment.structure
        self.logger = experiment.logger
        self.model_saver = experiment.train.model_saver

        # FIXME: Hack the save model path into logger path
        self.model_saver.dir_path = self.logger.save_dir(
            self.model_saver.dir_path)
        self.current_lr = 0

        self.total = 0

    def init_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(
            self.device, self.experiment.distributed, self.experiment.local_rank)
        return model

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.experiment.train.scheduler.learning_rate.get_learning_rate(
            epoch, step)

        for group in optimizer.param_groups:
            group['lr'] = lr
        self.current_lr = lr

    def train(self):
        self.logger.report_time('Start')
        self.logger.args(self.experiment)
        model = self.init_model()
        train_data_loader = self.experiment.train.data_loader
        if self.experiment.validation:
            validation_loaders = self.experiment.validation.data_loaders

        self.steps = 0

        model_paths = os.listdir(self.experiment.train.checkpoint.resume)

        for model_path in model_paths:
            self.experiment.train.checkpoint.resume = model_path
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

            optimizer = self.experiment.train.scheduler.create_optimizer(
                model.parameters())

            model.train()
            self.logger.info('Validating epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            val_loss = 0
            batch_no = 1

            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                if self.logger.verbose:
                    torch.cuda.synchronize()

                batch_loss = self.val_step(model, optimizer, batch,
                                           epoch=epoch, step=self.steps)

                self.logger.info("(validate) Epoch is [{}/{}], mini batch is [{}/{}], batch_loss is {:.8f}".format(
                epoch, self.experiment.train.epochs, batch_no, self.total, batch_loss))

                val_loss += batch_loss

                if self.logger.verbose:
                    torch.cuda.synchronize()

                batch_no += 1

            val_loss /= self.total

            self.logger.info("val_loss is {:.8f} (Epoch {})".format(val_loss, epoch))
            self.logger.info('Validation done {}'.format(model_path))

    def val_step(self, model, optimizer, batch, epoch, step, **kwards):
        results = model.forward(batch, training=True)

        if len(results) == 2:
            l, pred = results
            metrics = {}
        elif len(results) == 3:
            l, pred, metrics = results

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append('loss_{0}:{1:.4f}'.format(key, l_val.mean()))
        else:
            loss = l.mean()

        return loss

    def to_np(self, x):
        return x.cpu().data.numpy()
