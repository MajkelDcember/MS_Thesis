from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import logging
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "mistralai/Mistral-7B-v0.1"
quantized_model_dir = "mistralai/Mistral-7B-v0.1-4bit"

def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids':inp,'attention_mask': attention_mask})
    return traindataset, testenc

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
traindataset, _ = get_wikitext2(128, 0, 2048, pretrained_model_dir)
examples = traindataset

quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=False,
)

model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# Benchmarking
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=500,
    save_steps=500,
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=examples,
    eval_dataset=examples,
)

trainer.train()

# Plotting
train_metrics = trainer.evaluate()
train_loss = train_metrics["eval_loss"]
train_perplexity = 2**train_loss

fig, ax = plt.subplots()
ax.bar(["Train"], [train_perplexity], label="Perplexity")
ax.set_ylabel("Perplexity")
ax.set_title("Model Performance")
ax.legend()
plt.show()

model.quantize(examples)
model.save_quantized(quantized_model_dir)
