import torch
import math
from contextlib import suppress
import argparse
from ray import tune

from models.segmenter.factory import create_segmenter
from models.segmenter.config import model_cfg, opt_cfg, dataset_cfg
from models.segmenter.data.ade20k import ADE20KSegmentation
from models.segmenter.data.loader import Loader

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
    num_updates = epoch * len(data_loader)
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

        num_updates += 1

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
        "--epochs",
        type=float,
        help="max_epochs_per_job",
        required=True
    )

    args, _ = parser.parse_known_args()
    return args


def objective(config):
    if 'lr' in config.keys():
        opt_cfg['lr'] = config['lr']
    if 'momentum' in config.keys():
        opt_cfg['momentum'] = config['momentum']
    train_loader = create_dataset(dataset_kwargs=dataset_cfg)
    model_cfg['n_cls'] = train_loader.unwrapped.n_cls
    model = create_segmenter(model_cfg=model_cfg)
    optimizer = torch.optim.SGD(  # Tune the optimizer
        model.parameters(), lr=opt_cfg['lr'], momentum=opt_cfg['momentum']
    )

    for epoch in range(config['epochs']):
        # train for one epoch
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
        )
        tune.report(loss=loss)


def run_hpo():
    search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9), 'epochs': 25}

    # 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)


if __name__ == '__main__':
    run_hpo()
