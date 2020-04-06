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
        self.args = {"polygon": True,
                     "box_thresh": 0.5,
                     "result_dir": "./results"}

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
        validation_loaders = self.experiment.validation.data_loaders
        validation_loader = self.experiment.validation.data_loader["icdar2015"]

        self.steps = 0
        if self.experiment.train.checkpoint:
            self.experiment.train.checkpoint.restore_model(
                model, self.device, self.logger)
            epoch, iter_delta = self.experiment.train.checkpoint.restore_counter()
            self.steps = epoch * self.total + iter_delta

        optimizer = self.experiment.train.scheduler.create_optimizer(
            model.parameters())

        self.logger.report_time('Init')

        model.train()
        while True:
            self.logger.info('Training epoch ' + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(train_data_loader)

            train_loss = 0
            batch_no = 1

            for batch in train_data_loader:
                self.update_learning_rate(optimizer, epoch, self.steps)

                if self.logger.verbose:
                    torch.cuda.synchronize()

                batch_loss = self.train_step(model, optimizer, batch,
                                             epoch=epoch, step=self.steps)

                self.logger.info("Epoch is [{}/{}], mini batch is [{}/{}], batch_loss is {:.8f}".format(
                    epoch, self.experiment.train.epochs, batch_no, self.total, batch_loss))

                train_loss += batch_loss

                if self.logger.verbose:
                    torch.cuda.synchronize()

                self.steps += 1
                batch_no += 1
                self.logger.report_eta(self.steps, self.total, epoch)

            train_loss /= self.total

            self.logger.info("train_loss is {:.8f} (Epoch {})".format(train_loss, epoch))
            self.logger.info("<train>{},{:,8f}</train>".format(epoch, train_loss))

            self.logger.info("Saving model...")
            self.model_saver.save_checkpoint(model, 'dbnet_epoch{}'.format(epoch))

            self.logger.info("Validating...")
            val_loss = self.validate(validation_loaders, model, epoch)
            self.logger.info("val loss is {:.8f} (Epoch {})".format(val_loss, epoch))
            self.logger.info("<val>{},{:,8f}</val>".format(epoch, val_loss))

            self.logger.info("Evaluating...")

            self.logger.info("Epoch {}: F1-Score Results".format(epoch))
            os.system("CUDA_VISIBLE_DEVICES=0 python eval.py experiments/seg_detector/supermarket.yaml --resume pths/DB/db_epoch{} --polygon --box_thresh 0.5".format(epoch))

            epoch += 1

            if epoch > self.experiment.train.epochs:
                self.model_saver.save_checkpoint(model, 'final')
                break
            iter_delta = 0

    def train_step(self, model, optimizer, batch, epoch, step, **kwards):
        optimizer.zero_grad()

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
        loss.backward()
        optimizer.step()

        if step % self.experiment.logger.log_interval == 0:
            if isinstance(l, dict):
                line = '\t'.join(line)
                log_info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', '{}', 'lr:{:.4f}']).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info('step: %6d, epoch: %3d, loss: %.6f, lr: %f' % (
                    step, epoch, loss.item(), self.current_lr))
            self.logger.add_scalar('loss', loss, step)
            self.logger.add_scalar('learning_rate', self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric.mean(), step)
                self.logger.info('%s: %6f' % (name, metric.mean()))

            self.logger.report_time('Logging')

        return loss

    def eval_model(self, validation_loaders, model):
        model.eval()
        raw_metrics = list()

        with torch.no_grad():
            for name, data_loader in validation_loaders.items():
                for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                    pred = model.forward(batch, training=False)
                    output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    self.format_output(batch, output)
                    raw_metric = self.structure.measurer.validate_measure(batch, output, is_output_polygon=self.args['polygon'], box_thresh=self.args['box_thresh'])
                    raw_metrics.append(raw_metric)

        metrics = self.structure.measurer.gather_measure(raw_metrics, self.logger)
        model.train()

        return metrics

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def validate(self, validation_loaders, model, epoch):
        val_loss = 0
        
        for name, data_loader in validation_loaders.items():
            num_val = len(data_loader)
            batch_no = 1

            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                results = model.forward(batch, training=True)

                if len(results) == 2:
                    l, pred = results
                    metrics = {}
                elif len(results) == 3:
                    l, pred, metrics = results

                if isinstance(l, dict):
                    loss = torch.tensor(0.).cuda()
                    for key, l_val in l.items():
                        loss += l_val.mean()
                else:
                    loss = l.mean()

                batch_loss = loss
                self.logger.info("(validation) Epoch is [{}/{}], mini batch is [{}/{}], batch_loss is {:.8f}".format(
                    epoch, self.experiment.train.epochs, batch_no, num_val, batch_loss))
                
                val_loss += batch_loss
                batch_no += 1

            val_loss /= num_val
            self.logger.info("val_loss is {:.8f} (Epoch {})".format(train_loss, epoch))

        return val_loss
    
    def to_np(self, x):
        return x.cpu().data.numpy()
