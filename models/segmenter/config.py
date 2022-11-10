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
    "image_size": (160, 320),
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

dataset_dir = lambda: '/home/mobileye/isaacd/datasets/'
