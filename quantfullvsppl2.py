from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from data import get_loaders 
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Seed setting function
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "meta-llama/CodeLlama-7b-hf"

# Load model and tokenizer once
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and prepare dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = dataset.shuffle(seed=42).select(range(10))['text']
encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings).input_ids.to(device)

# Perplexity evaluation function

seqlen = model.config.max_position_embeddings
# Function to evaluate perplexity (ppl) on a specific dataset
def eval_ppl(model, tokenizer, device=torch.device('cpu')):
    # Set dataset
    dataset = "wikitext2"

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    testloader = get_loaders(dataset, seed=0, seqlen=seqlen, tokenizer=tokenizer)

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext_train(model, testloader)
    return ppl_test

# Function to evaluate perplexity (ppl) specifically on wikitext training data

def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []

    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0, nsamples, bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)


    # Convert mean negative log likelihood to perplexity
    perplexity = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    print(f"Perplexity: {perplexity}")

    return perplexity






unquantized_ppl = eval_ppl(model, tokenizer)
print(f"Unquantized Perplexity: {unquantized_ppl}")

# Define quantization and evaluation loop
def quantize_weights(layer, v):
    with torch.no_grad():
        quantized_weights = torch.round(layer.weight * v) 
        quantized_weights = torch.clamp(quantized_weights, -8, 7)
        quantized_weights = quantized_weights/ v
        layer.weight.copy_(quantized_weights)

def quantize_all_linear_layers(model, v):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            v = 32767/torch.mean(module.weight)
            quantize_weights(module, v)

vs = [2**i for i in range(25)]
perplexities = []
del model  # Ensure clean up to prevent memory issues
torch.cuda.empty_cache()
for v in vs:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    quantize_all_linear_layers(model, v)
    ppl = eval_ppl(model, tokenizer)
    perplexities.append(ppl)
    print(f"Quantization v={v}, Perplexity: {ppl}")
    del model  # Ensure clean up to prevent memory issues
    torch.cuda.empty_cache()

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(vs, perplexities, marker='o')
plt.axhline(y=unquantized_ppl, color='r', linestyle='--', label=f'Unquantized PPL: {unquantized_ppl}')
plt.xscale('log')
plt.xlabel('Quantization Level v')
plt.ylabel('Perplexity')
plt.title('Perplexity vs. Quantization Level (Wikitext)')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig('perplexity_vs_quantization.png')
plt.show()
