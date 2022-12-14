import os

model_cfg = {
    "backbone": "hi",
    "decoder": {
        "name": "mask_transformer",
        "n_layers": 2,
        "drop_path_rate": 0.0,
        "dropout": 0.1,
        "mup": False,
        "attn_mult": None
    },
    "normalization": None,
    "d_model": 192,
    "image_size": (512, 512),
    "n_layers": 12,
    "n_heads": 3,
    "patch_size": 16,
    "mup": False,
    'attn_mult': None,
}

opt_cfg = {
    "lr": 0.001,
    "weight_decay": 0,
    "momentum": 0.9,
}

dataset_cfg = {
    'image_size': 512,
    'crop_size': 512,
    'batch_size': 8,
    'split': 'train',
    'normalization': 'vit',
    'num_workers': 10,
}

LOCAL_LOC, S3_LOC = '', ''  # local dataset location and aws dataset location
dataset_dir = lambda: os.environ[
    "SM_CHANNEL_DATA"] if "TRAINING_JOB_NAME" in os.environ else LOCAL_LOC  # using FastFileMode in AWS
inputs = {'DATA': S3_LOC}
