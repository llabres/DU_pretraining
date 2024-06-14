import os
import copy
import torch
import transformers

from transformers import get_scheduler


def build_optimizer(model, length_train_loader, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']))
    num_training_steps = config['train_epochs'] * length_train_loader
    if 'warmup_iterations' in config.keys():
        num_warmup_steps = config['warmup_iterations']
    elif 'warmup_ratio' in config.keys():
        num_warmup_steps = int(config['warmup_ratio'] * num_training_steps)
    else:
        num_warmup_steps = 0

    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, lr_scheduler


def build_model(config):

    available_models = ['vt5', 'mp-vt5']
    if config['model_name'].lower() == 'vt5':
        from models.VT5.vt5 import VT5
        from transformers import T5Config
        model_config = T5Config.from_pretrained(config['model_weights'])
        model_config.pretraining = bool(config.get('pretraining', 0))
        model_config.continuous_spatial_embeddings = bool(config.get('continuous_spatial_embeddings', 0))
        model_config.max_patches = config.get('max_patches', 196)
        model_config.page_prediction = bool(config.get('page_prediction', 0))
        model_config.feature_extractor_name = 'google/pix2struct-base'
        model_config.image_resolution = config.get('image_resolution', 512) # in patches
        model = VT5.from_pretrained(config['model_weights'], config=model_config)
    
    elif config['model_name'].lower() == 'mp-vt5':
        from transformers import T5Config
        from models.MP_VT5.mp_vt5 import MP_VT5
        model_config = T5Config.from_pretrained(config['model_weights'])
        model_config.max_pages = config.get('max_pages', 2)
        model_config.n_page_tokens = config.get('n_page_tokens', 10)
        model_config.use_all_tokens = bool(config.get('use_all_tokens', 0))
        model_config.continuous_spatial_embeddings = bool(config.get('continuous_spatial_embeddings', 0))
        model_config.pretraining = bool(config.get('pretraining', 0))
        model_config.page_prediction = bool(config.get('page_prediction', 0))
        model_config.max_patches = config.get('max_patches', 196)
        model_config.feature_extractor_name = 'google/pix2struct-base'
        model_config.pixel_only = bool(config.get('pixel_only', 0))
        model_config.question_in_all_pages = bool(config.get('question_in_all_pages', 0))
        model_config.image_resolution = config.get('image_resolution', 512) # in patches
    
        model = MP_VT5.from_pretrained(config['model_weights'], config=model_config, ignore_mismatched_sizes=True)
    else:
        raise ValueError(f"Value '{config['model_name']}' for model selection not expected. Please choose one of {', '.join(available_models)}")
    
    model_config = copy.deepcopy(model.config)
    config['model_config'] = model_config.__dict__
    if 'vision_config' in model_config.__dict__.keys():
        config['model_config']['vision_config'] = model_config.vision_config.__dict__
    if 'text_config' in model_config.__dict__.keys():
        config['model_config']['text_config'] = model_config.text_config.__dict__

    return model, config


def build_dataset(config, split):
    config['split'] = split

    if config['model_name'].lower() in ['vt5', 'mp-vt5']:
        config['use_ocr'] = True

    if config['model_name'].lower() in ['vt5', 'mp-vt5']:
        config['use_images'] = True

    config['page_prediction'] = bool(config.get('page_prediction', 0))
    
    # Build dataset
    if config['dataset_name'].lower() == 'sp-idl':
        if config['split'] == 'test' or config['split'] == 'val':
            from my_datasets import SingleDocVQA
            import yaml
            dataset_config = yaml.safe_load(open("configs/datasets/SingleDocVQA.yml", "r"))
            dataset_config['split'] = config['split']
            dataset_config['use_images'] = config['use_images'] if 'use_images' in config else False
            dataset_config['use_ocr'] = config['use_ocr'] if 'use_ocr' in config else False
            dataset_config['gt_answers'] = True
            dataset = SingleDocVQA(dataset_config)
        else:
            from my_datasets import SP_IDL
            dataset = SP_IDL(config)
    else:
        raise ValueError

    return dataset
