import json
import os.path
import warnings

import torch

from datasets.custom_dataset import CustomDataloader
from loss import get_loss_fn
from optimizer import get_optimizer, get_scheduler
from train.trainer import Trainer
from net.custom_transformer import Transformer

warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)


def training():
    batch_size = 64
    total_epoch = 10
    lr = 0.00005
    hidden_size = 768
    num_attention_heads = 8
    dropout = 0.2
    num_layers = 6
    is_training = True

    root_dir = os.path.dirname(__file__)
    data_dir = os.path.join(root_dir, '../datas')
    output_dir = os.path.join(root_dir, '../output/custom_transformer')
    os.makedirs(output_dir, exist_ok=True)

    example_input = (
        torch.randint(0, 100, size=(4, 10)).to(dtype=torch.long),
        torch.ones(size=(4, 10)).to(dtype=torch.float32),
        torch.randint(0, 100, size=(4, 5)).to(dtype=torch.long),
    )

    custom_dataloader = CustomDataloader(batch_size=batch_size, data_dir=data_dir)
    train_dataloader, test_dataloader, token2idx = custom_dataloader.get_dataloader()

    vocab_size = len(token2idx)
    net = Transformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        num_layers=num_layers,
        is_training=is_training
    )

    net_configs = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_attention_heads': num_attention_heads,
        'dropout': dropout,
        'num_layers': num_layers,
        'is_training': is_training
    }

    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as writer:
        json.dump(net_configs, writer, indent=4)
    with open(os.path.join(output_dir, 'vocab.json'), 'w', encoding='utf-8') as writer:
        json.dump(token2idx, writer, ensure_ascii=False, indent=4)

    loss_fn = get_loss_fn()
    optimizer = get_optimizer(net, lr, 'adamw')
    # scheduler = get_scheduler(optimizer, 'linear')
    scheduler = get_scheduler(optimizer, 'steplr')
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
    )
    trainer.fit()


if __name__ == '__main__':
    training()
