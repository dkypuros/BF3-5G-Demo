# File: cached_ttm_eval.py

import os
import math
import tempfile
import torch
import time
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
import numpy as np
import pandas as pd
from tsfm_public.models.tinytimemixer.utils import count_parameters, get_data, plot_preds
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.callbacks import TrackingCallback
import warnings
from enhanced_json_cache import EnhancedJSONCache

warnings.filterwarnings("ignore")

SEED = 42
set_seed(SEED)

target_dataset = "ettm2"
DATA_ROOT_PATH = "/dccstor/tsfm23/datasets/"
OUT_DIR = "ttm_finetuned_models/"
TTM_MODEL_NAME = "ibm-granite/granite-timeseries-ttm-v1"

# Initialize cache
cache = EnhancedJSONCache('ttm_cache.json', expiry_time=86400, max_size=1000)  # 24 hour expiry

def get_cached_data(dataset_name, context_length, forecast_length, fewshot_fraction, data_root_path):
    cache_key = f"{dataset_name}_{context_length}_{forecast_length}_{fewshot_fraction}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data['train'], cached_data['val'], cached_data['test']
    
    dset_train, dset_val, dset_test = get_data(dataset_name, context_length, forecast_length, fewshot_fraction, data_root_path)
    cache.set(cache_key, {'train': dset_train, 'val': dset_val, 'test': dset_test})
    return dset_train, dset_val, dset_test

def zeroshot_eval(dataset_name, batch_size, context_length=512, forecast_length=96, prediction_filter_length=None):
    cache_key = f"zeroshot_{dataset_name}_{context_length}_{forecast_length}_{prediction_filter_length}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print("Using cached zero-shot evaluation result")
        print(cached_result)
        return

    _, _, dset_test = get_cached_data(dataset_name, context_length, forecast_length, 1.0, DATA_ROOT_PATH)
    
    if prediction_filter_length is None:
        zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME)
    else:
        if prediction_filter_length <= forecast_length:
            zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME, prediction_filter_length=prediction_filter_length)
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")

    temp_dir = tempfile.mkdtemp()
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(output_dir=temp_dir, per_device_eval_batch_size=batch_size,)
    )

    print("+" * 20, "Test MSE zero-shot", "+" * 20)
    zeroshot_output = zeroshot_trainer.evaluate(dset_test)
    print(zeroshot_output)

    cache.set(cache_key, zeroshot_output)

    plot_preds(trainer=zeroshot_trainer, dset=dset_test, plot_dir=os.path.join(OUT_DIR, dataset_name), plot_prefix="test_zeroshot", channel=0)

def fewshot_finetune_eval(dataset_name, batch_size, learning_rate=0.001, context_length=512, forecast_length=96,
                          fewshot_percent=5, freeze_backbone=True, num_epochs=50, save_dir=OUT_DIR, prediction_filter_length=None):
    
    cache_key = f"fewshot_{dataset_name}_{context_length}_{forecast_length}_{fewshot_percent}_{freeze_backbone}_{num_epochs}_{prediction_filter_length}"
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        print(f"Using cached few-shot {fewshot_percent}% evaluation result")
        print(cached_result)
        return

    out_dir = os.path.join(save_dir, dataset_name)
    print("-" * 20, f"Running few-shot {fewshot_percent}%", "-" * 20)
    
    dset_train, dset_val, dset_test = get_cached_data(dataset_name, context_length, forecast_length, fewshot_percent / 100, DATA_ROOT_PATH)

    if "ett" in dataset_name:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME, head_dropout=0.7)
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME, head_dropout=0.7, prediction_filter_length=prediction_filter_length)
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")
    else:
        if prediction_filter_length is None:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME)
        elif prediction_filter_length <= forecast_length:
            finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(TTM_MODEL_NAME, prediction_filter_length=prediction_filter_length)
        else:
            raise ValueError(f"`prediction_filter_length` should be <= `forecast_length")

    if freeze_backbone:
        print("Number of params before freezing backbone", count_parameters(finetune_forecast_model))
        for param in finetune_forecast_model.backbone.parameters():
            param.requires_grad = False
        print("Number of params after freezing the backbone", count_parameters(finetune_forecast_model))

    print(f"Using learning rate = {learning_rate}")
    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(out_dir, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=8,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(out_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.0,)
    tracking_callback = TrackingCallback()

    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(optimizer, learning_rate, epochs=num_epochs, steps_per_epoch=math.ceil(len(dset_train) / (batch_size)),)

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=dset_train,
        eval_dataset=dset_val,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    finetune_forecast_trainer.train()

    print("+" * 20, f"Test MSE after few-shot {fewshot_percent}% fine-tuning", "+" * 20)
    fewshot_output = finetune_forecast_trainer.evaluate(dset_test)
    print(fewshot_output)
    print("+" * 60)

    cache.set(cache_key, fewshot_output)

    plot_preds(trainer=finetune_forecast_trainer, dset=dset_test, plot_dir=os.path.join(OUT_DIR, dataset_name), plot_prefix="test_fewshot", channel=0)

if __name__ == "__main__":
    zeroshot_eval(dataset_name=target_dataset, batch_size=64)
    fewshot_finetune_eval(dataset_name=target_dataset, batch_size=64)