import sys
import os.path as o
sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), "..")))

from directories import *
import argparse
import logging

import pickle

# from utils.task_loaders import get_best_epochs

### Parse inputs
parser = argparse.ArgumentParser()

parser.add_argument("--TASK")
parser.add_argument("--SPLIT", default="train")
parser.add_argument("--MAX_LENGTH", default=128, type=int)
parser.add_argument("--BATCH_SIZE", default=4, type=int)
parser.add_argument("--MULTI_GPU", action="store_true")
parser.add_argument("--DATA_SEED", default=22, type=int)
parser.add_argument("--GPU", default=1, type=int)
args = parser.parse_args()

# Hyper Params
BACKBONE = "bert"
SELECTED_GPU = "multi" if args.MULTI_GPU else args.GPU
TASK = args.TASK
SPLIT = args.SPLIT
MAX_LENGTH = args.MAX_LENGTH
MODEL_PATH = 'bert-base-uncased' if MAX_LENGTH <= 512 else LONG_BERT_DIR
SAL_BATCH_SIZE = args.BATCH_SIZE
SEED = args.DATA_SEED
LOAD_MODEL_PATH = WORKER_MODEL_PATH(TASK, BACKBONE, seed=SEED)
BEST_EPOCH = 1# get_best_epochs(f"./directory/bert/{SEED}/{TASK}/logs/ft_logs.json")[-3:] + 1
SAL_AGG = "L2"
SAL_NORM = None
SALIENCIES_PATH = WORKER_SALIENCIES_PATH_to_WRITE(TASK, BACKBONE, SPLIT, seed=SEED)

# Import Requirements
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from transformers import (
    BertConfig,
    BertTokenizer, 
)

from modeling.modeling_tf_bert import TFBertForSequenceClassification
from utils.task_loaders import load_task


if SELECTED_GPU == "multi":
  mirrored_strategy = tf.distribute.MirroredStrategy()
else:
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.set_logical_device_configuration(
        gpus[int(SELECTED_GPU)],
        [tf.config.LogicalDeviceConfiguration(memory_limit=7100)])
      tf.config.experimental.set_visible_devices(gpus[int(SELECTED_GPU)], 'GPU')
      # tf.config.experimental.set_memory_growth(gpus[int(SELECTED_GPU)], True)
      # tf.config.experimental.se.set_per_process_memory_fraction(0.92)
      print(gpus[int(SELECTED_GPU)])
    except RuntimeError as e:
      print(e)
  else:
    print('GPU not found!')


# Load Tokenizer & Dataset
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

### Load Task
datasets, info, metrics = load_task[TASK](tokenizer=tokenizer, max_length=MAX_LENGTH, training_batch_size=SAL_BATCH_SIZE, eval_batch_size=SAL_BATCH_SIZE, seed=SEED)

train_steps = info[SPLIT + "_steps"]
num_labels = info["num_labels"]
num_train_examples = info[SPLIT + "_examples"] #+ SAL_BATCH_SIZE

if SPLIT == "val":
  first_key = list(datasets["evals"].keys())[0]
  train_dataset = datasets["evals"][first_key]
elif SPLIT == "test":
  first_key = list(datasets["tests"].keys())[0]
  train_dataset = datasets["tests"][first_key]
else:
  train_dataset = datasets[SPLIT]

train_dataset = train_dataset.unbatch().repeat(-1).take((np.math.ceil(num_train_examples / SAL_BATCH_SIZE)) * SAL_BATCH_SIZE).batch(SAL_BATCH_SIZE)
if SELECTED_GPU == "multi":
  train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

config = BertConfig.from_pretrained(MODEL_PATH, num_labels=num_labels)

@tf.function
def get_saliency(model, inputs, training, saliency_agg, norm):
    return model.saliency_step(inputs, training, saliency_agg, norm)['saliencies']

@tf.function
def distributed_get_saliency(model, inputs, training, saliency_agg, norm):
  per_replica_sals = mirrored_strategy.run(get_saliency, args=(model, inputs, training, saliency_agg, norm))
  return mirrored_strategy.gather(per_replica_sals, axis=0)

sal_mat = np.zeros((num_train_examples, MAX_LENGTH), dtype=np.float32)
sal_mat_dict = {}
# Loop on best epochs
print("start of epochs", flush=True)
for epoch in [10]:
    # Load Model
    if SELECTED_GPU == "multi":
      with mirrored_strategy.scope():
        model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
        model.load_weights(LOAD_MODEL_PATH+str(epoch)+'.h5')
        model.to_AdapLeR()
        model.bert.encoder._lambda.assign(1e+10)
        model.bert.encoder.ETA.assign(np.ones(12) * 1e-10)
    else:
      model = TFBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config, from_pt=MAX_LENGTH > 512)
      model.load_weights(LOAD_MODEL_PATH+str(epoch)+'.h5')
      model.to_AdapLeR()
      model.bert.encoder._lambda.assign(1e+10)
      model.bert.encoder.ETA.assign(np.ones(12) * 1e-10)

    print("start of sals", flush=True)
    # Compute Saliencies
    pbar = tqdm(enumerate(train_dataset), total=train_steps, leave=True)
    idx = 0
    for step, inputs in pbar:
        if SELECTED_GPU == "multi":
          outputs = distributed_get_saliency(model, inputs, training=False, saliency_agg=SAL_AGG, norm=SAL_NORM)
        else:
          outputs = get_saliency(model, inputs, training=False, saliency_agg=SAL_AGG, norm=SAL_NORM)
        
        for j in range(len(outputs)):
          sal_mat_dict[idx] = outputs[j].numpy()
          idx += 1

    sal_mat = np.zeros((len(sal_mat_dict), MAX_LENGTH), dtype=np.float32)
    for key in sal_mat_dict:
      sal_mat[key] = sal_mat_dict[key]

    with open(SALIENCIES_PATH+f"{TASK}_lot_grads.npy", 'wb') as f:
      np.save(f, sal_mat)
