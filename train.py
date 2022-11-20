import torch
import math
from contextlib import suppress
import argparse

from models.segmenter.factory import create_segmenter
from models.segmenter.config import model_cfg, opt_cfg, dataset_cfg
from models.segmenter.data.ade20k import ADE20KSegmentation
from models.segmenter.data.loader import Loader
from syne_tune.report import Reporter

IGNORE_LABEL = 255


def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    # load dataset_name
    dataset = ADE20KSegmentation(split=split, **dataset_kwargs)

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=False,
        split=split,
    )
    return dataset


def train_one_epoch(
        model,
        data_loader,
        optimizer,
        epoch,
        amp_autocast=suppress,
        loss_scaler=None,
        device=None
):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    model.train()
    data_loader.set_epoch(epoch)
    for batch in data_loader:
        im = batch['im']
        if device:
            im = batch["im"].to(device)
        seg_gt = batch["segmentation"].long()
        if device:
            seg_gt = seg_gt.to(device)

        with amp_autocast():
            seg_pred = model.forward(im)
            loss = criterion(seg_pred, seg_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            optimizer.step()

        if device:
            torch.cuda.synchronize()

    return loss.item()


def parse_args():
    parser = argparse.ArgumentParser(description="HPO script")
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="weight_decay"
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="momentum"
    )
    parser.add_argument(
        "--mup",
        action="store_true",
        help="momentum"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="max_epochs_per_job",
        required=True
    )
    parser.add_argument(
        "--dim",
        type=int,
        help="dimensions of model",
        required=True
    )

    args, _ = parser.parse_known_args()
    return vars(args)


def objective(config):
    if 'lr' in config.keys():
        opt_cfg['lr'] = config['lr']

    if 'momentum' in config.keys():
        opt_cfg['momentum'] = config['momentum']

    if 'mup' in config.keys():
        if config['mup']:
            model_cfg['mup'] = True
            model_cfg['attn_mult'] = 8
    if 'dim' in config.keys():
        model_cfg['d_model'] = config['dim']
    train_loader = create_dataset(dataset_kwargs=dataset_cfg)
    model_cfg['n_cls'] = train_loader.unwrapped.n_cls
    print(model_cfg)
    model = create_segmenter(model_cfg=model_cfg)
    device = 'cuda'
    if device:
        model.to(device)
    optimizer = torch.optim.SGD(  # Tune the optimizer
        model.parameters(), lr=opt_cfg['lr'], momentum=opt_cfg['momentum']
    )
    report = Reporter(add_cost=False)
    for epoch in range(1, config['epochs'] + 1):
        # train for one epoch
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            device=device
        )
        report(epoch=epoch, loss=loss )



if __name__ == '__main__':
    args = parse_args()
    objective(args)
