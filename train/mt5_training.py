import os
import os.path
import warnings
import sys

root_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(root_dir, '../'))

import torch

from datasets.mt5_dataset import MT5Dataloader
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler
from train.trainer import Trainer
from net.mt5_net import MT5BaseNet

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def training():
    batch_size = 64
    total_epoch = 10
    lr = 0.0005

    # mt5_path = r'/home/featurize/data/t5-pegasus-small'
    mt5_path = r'C:\Users\du\.cache\huggingface\hub\hub\t5-pegasus-small'
    data_dir = os.path.join(root_dir, '../datas')
    output_dir = os.path.join(root_dir, '../output/t5-pegasus')
    os.makedirs(output_dir, exist_ok=True)

    example_input = (
        torch.randint(0, 100, size=(4, 10)).to(dtype=torch.long),
        torch.ones(size=(4, 10)).to(dtype=torch.float32),
        torch.randint(0, 100, size=(4, 5)).to(dtype=torch.long),
    )

    custom_dataloader = MT5Dataloader(batch_size=batch_size, data_dir=data_dir, t5_model_dir=mt5_path)
    train_dataloader, test_dataloader, vocab_size = custom_dataloader.get_dataloader()

    net = MT5BaseNet(model_path=mt5_path)

    loss_fn = get_loss_fn()
    optimizer = get_optimizer(net, lr, 'AdamW')
    scheduler = get_scheduler(optimizer, 'StepLr')
    # scheduler = None
    trainer = Trainer(
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optim=optimizer,
        scheduler=scheduler,
        total_epoch=total_epoch,
        output_dir=output_dir,
        example_input=example_input,
        eval_metrics=None,
        early_stop=True,
        early_stop_step=5,
        device='cuda'
        # device='cpu'
    )
    trainer.fit()


if __name__ == '__main__':
    training()
