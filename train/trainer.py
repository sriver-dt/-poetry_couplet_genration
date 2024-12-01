import atexit
import logging
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Callable, Union

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, net: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_fn: Callable, optim: Optimizer, scheduler: LRScheduler, total_epoch: int,
                 output_dir: Union[str, PathLike], example_input, eval_metrics: Callable = None,
                 early_stop: bool = True, early_stop_step: int = 5, device: str = 'cpu'
                 ):
        super(Trainer, self).__init__()
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        self.device = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        self.total_epoch = total_epoch
        self.eval_metric = eval_metrics
        self.best_score = 0
        self.start_epoch = 0
        self.save_model_dir = Path(output_dir)
        self.early_stop = early_stop
        self.early_stop_step = early_stop_step

        # 可视化
        str_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.summary_writer = SummaryWriter(log_dir=str(self.save_model_dir.joinpath(f'summary/{str_time}')))
        self.summary_writer.add_graph(self.net, input_to_model=example_input)
        self.global_step = 0
        atexit.register(self.close)

        # 模型恢复
        model_path_dict = {}
        if not self.save_model_dir.exists():
            self.save_model_dir.mkdir(parents=True)
        if self.save_model_dir.exists() and not self.is_empty(self.save_model_dir):
            model_path_dict = {model_path.parts[-1]: model_path for model_path in self.save_model_dir.iterdir()}
        if 'best.pkl' in model_path_dict.keys():
            model_path = model_path_dict['best.pkl']
        elif 'last.pkl' in model_path_dict.keys():
            model_path = model_path_dict['last.pkl']
        else:
            model_path = None
        if model_path is not None:
            model_state_dict = torch.load(f=model_path, map_location=torch.device('cpu'))
            missing_keys, unexpected_keys = self.net.load_state_dict(model_state_dict['model_state'])
            self.best_score = model_state_dict['best_score']
            self.start_epoch = model_state_dict['epoch'] + 1
            self.total_epoch += self.start_epoch
            print('正在进行模型恢复')
            print(f'missing_keys: {missing_keys}')
            print(f'unexpected_keys : {unexpected_keys}')

    def close(self):
        logging.info("close resources....")
        self.summary_writer.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __enter__(self):
        return self

    @staticmethod
    def is_empty(path: Path):
        try:
            next(path.iterdir())
            return False
        except StopIteration:
            return True

    def fit(self):
        early_stop_count, best_epoch = 0, 0
        self.net.to(device=self.device)
        for epoch in range(self.start_epoch, self.total_epoch):
            self.train(epoch)
            current_score = self.eval(epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            self.save(epoch=epoch, name='last')  # 保存最后一个模型
            if current_score >= self.best_score:
                best_epoch = epoch
                self.best_score = current_score
                self.save(epoch=epoch, name='best')  # 保存最好的模型
                early_stop_count = 0
                continue
            early_stop_count += 1
            if early_stop_count == self.early_stop_step and self.early_stop:
                logging.info(f'提前停止：-----best_epoch {best_epoch}  -----best_score {self.best_score}')
                break

    def train(self, epoch):
        self.net.train()

        pbar = tqdm(range(len(self.train_dataloader)), desc='training')
        for batch_idx, ((batch_input, batch_mask, batch_dec_input, batch_dec_mask), batch_label) in enumerate(
                self.train_dataloader):
            self.global_step += 1
            batch_input = batch_input.to(device=self.device)
            batch_mask = batch_mask.to(device=self.device)
            batch_dec_input = batch_dec_input.to(device=self.device) if batch_dec_input is not None else None
            batch_dec_mask = batch_dec_mask.to(device=self.device) if batch_dec_mask is not None else None
            batch_label = batch_label.to(device=self.device)
            output = self.net(
                input_ids=batch_input,
                decoder_input_ids=batch_dec_input,
                attention_mask=batch_mask,
                decoder_attention_mask=batch_dec_mask
            )
            self.optim.zero_grad()
            # 反向传播
            loss = self.loss_fn(torch.permute(output, dims=(0, 2, 1)), batch_label)
            loss.backward()
            self.optim.step()

            output = torch.softmax(output, dim=-1)
            pred = torch.argmax(output, dim=-1)
            if self.eval_metric is not None:
                score = self.eval_metric(batch_label.cpu().numpy(), pred.cpu().numpy())
            else:
                score = pred.eq(batch_label).sum().item() / torch.numel(batch_label)

            # 进度条信息
            pbar.set_description(f'train epoch:{epoch + 1}/{self.total_epoch}')
            pbar.set_postfix(
                loss=round(loss.cpu().item(), 5),
                acc=round(score, 5),
            )
            pbar.update(1)

            # 可视化
            self.summary_writer.add_scalar('loss', loss.cpu().item(), global_step=self.global_step)
            self.summary_writer.add_scalar('train_score', score, global_step=self.global_step)

    def eval(self, epoch):
        self.net.eval()

        pbar = tqdm(range(len(self.test_dataloader)), desc='eval')

        corrects = 0
        trues = 0
        with torch.no_grad():
            for (batch_input, batch_mask, batch_dec_input, batch_dec_mask), batch_label in self.test_dataloader:
                batch_input = batch_input.to(device=self.device)
                batch_mask = batch_mask.to(device=self.device)
                batch_dec_input = batch_dec_input.to(device=self.device) if batch_dec_input is not None else None
                batch_dec_mask = batch_dec_mask.to(device=self.device) if batch_dec_mask is not None else None
                output = self.net(
                    input_ids=batch_input,
                    decoder_input_ids=batch_dec_input,
                    attention_mask=batch_mask,
                    decoder_attention_mask=batch_dec_mask
                )

                output = torch.softmax(output, dim=-1)
                pred = torch.argmax(output, dim=-1).cpu()
                corrects += pred.eq(batch_label).sum().item()
                trues += torch.numel(batch_label)

                pbar.update(1)

            score = corrects / trues

            print(f'eval epoch: {epoch} score: {score:.4f}')

            self.summary_writer.add_scalar('eval_score', score, global_step=epoch)
        return score

    def save(self, epoch, name):
        state_dict = {
            'epoch': epoch,
            'best_score': self.best_score,
            'model_state': self.net.state_dict()
        }
        torch.save(state_dict, f=self.save_model_dir.joinpath(f'{name}.pkl'))
