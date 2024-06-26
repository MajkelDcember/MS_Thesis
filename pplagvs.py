from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from datasets import load_dataset

import matplotlib.pyplot as plt


import numpy as np
import random
import os

def set_seed(seed=42):
    """Fixes random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # If you are using cudnn, add these
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call this function at the very start of your script
set_seed(1)


# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Ensure the tokenizer has a padding token set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Correct path to access and modify a specific layer
layer = model.model.layers[3].mlp.gate_proj

# Initialize an activation dictionary to store layer outputs
activation = {}

# Define a function to register a hook that captures layer activations
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Register the forward hook to the correct layer
layer.register_forward_hook(get_activation('mlp.gate_proj'))

# Optionally, add normal noise to the weights
def quantize_weights(layer, v):
    """
    Applies integer quantization to the weights of a layer.

    Parameters:
    - layer: The layer whose weights will be quantized.
    - v: A scaling factor controlling the level of quantization.
    """
    with torch.no_grad():
        # Scale the weights by v and round to the nearest integer
        quantized_weights = torch.round(v * layer.weight)
        # Scale back the quantized weights
        layer.weight.copy_(quantized_weights / v)

# Choose a scaling factor for quantization

print("Before quantization:", layer.weight)



# Load and preprocess dataset
wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
# texts = wikitext_dataset.shuffle(seed=42).select(range(10))['text']
texts = wikitext_dataset['text']

# Preprocess and encode texts
encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_position_embeddings)
encoded = encoded.input_ids.to(device)

# Define the evaluation function for perplexity

def eval_ppl_wikitext(model, encoded, bs=1):
    nsamples = encoded.size(0)
    nlls = []

    for i in range(0, nsamples, bs):
        inputs = encoded[i:i+bs]
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            neg_log_likelihood = loss * inputs.size(1)
            nlls.append(neg_log_likelihood)

    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / (nsamples * model.config.max_position_embeddings))
    return ppl.item()
unquantized_ppl = eval_ppl_wikitext(model, encoded)
print("Unquantized Perplexity:", unquantized_ppl)

vs = [ 1, 10, 100, 1000, 10000]
perplexities = []
del model  # Delete the model
torch.cuda.empty_cache()
for v in vs:
    # Re-initialize the model to reset weights
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Access and quantize the specific layer
    layer = model.model.layers[3].mlp.gate_proj
    with torch.no_grad():
        quantized_weights = torch.round(v * layer.weight) / v
        layer.weight.copy_(quantized_weights)
    
    # Calculate perplexity
    ppl = eval_ppl_wikitext(model, encoded)
    perplexities.append(ppl)
    print(f"After quantization with level v={v}:", layer.weight)
    print(f"Quantization v={v}, Perplexity: {ppl}")
    del model  # Delete the model
    torch.cuda.empty_cache()



# Plotting
plt.plot( vs, perplexities, marker='o')  
plt.axhline(y=unquantized_ppl, color='r', linestyle='--', label=f'unquantized_ppl: {unquantized_ppl}')

# Plotting
plt.plot(vs, perplexities, marker='o')
plt.xscale('log')
plt.xlabel('Quantization Level v')
plt.ylabel('Perplexity')
plt.title('Perplexity vs. Quantization Level')
plt.grid(True, which="both", ls="--")

plt.savefig('perplexity_vs_quantization2.png')

plt.show()

