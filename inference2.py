from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
import torch
import gc 
from tqdm import tqdm
import random
import numpy as np

# Set up the prompt and model directories
Prompt = "Here is a funny joke: "
max_tokenz = 100
pretrained_model_dir = "mistralai/Mistral-7B-v0.1"
quantized_model_dir_4bit = "mistralai/Mistral-7B-v0.1-4bit"

# Load the tokenizer and the quantized model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
model_4bit = AutoGPTQForCausalLM.from_quantized(quantized_model_dir_4bit, device="cuda:0")

# Generate text with the quantized model
pipeline = TextGenerationPipeline(model=model_4bit, tokenizer=tokenizer)
quantized_output_4bit = pipeline(Prompt, max_new_tokens=max_tokenz)[0]["generated_text"]
print("Quantized model 4-bit output:")
print(quantized_output_4bit)

# Load the dataset
from datasets import load_dataset




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
trainenc, testenc = get_wikitext2(128, 0, 2048, pretrained_model_dir)


# Set up the training arguments
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=0,  # Set to 0 to skip training
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=500,
    save_steps=500,
    seed=42,
)

# Create the trainer and evaluate the model
trainer = Trainer(
    model=model_4bit,
    args=training_args,
    train_dataset=trainenc,
    eval_dataset=testenc,
)

# Add progress bar for evaluation
with tqdm(total=1, desc="Evaluating model") as pbar:
    trainer.evaluate()
    pbar.update(1)

# Plot the results
import matplotlib.pyplot as plt
train_metrics = trainer.evaluate()
train_loss = train_metrics["eval_loss"]
train_perplexity = 2**train_loss

fig, ax = plt.subplots()
ax.bar(["Train"], [train_perplexity], label="Perplexity")
ax.set_ylabel("Perplexity")
ax.set_title("Model Performance")
ax.legend()
plt.show()