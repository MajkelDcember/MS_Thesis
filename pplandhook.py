from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
from datasets import load_dataset

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
v = 1  # Example scaling factor, adjust as needed
print("Before quantization:", layer.weight)

# Apply integer quantization to the specified layer
quantize_weights(layer, v)
print("After quantization:", layer.weight)


# Load and preprocess dataset
wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
texts = wikitext_dataset.shuffle(seed=42).select(range(10))['text']

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

# Evaluate and print overall perplexity
overall_perplexity = eval_ppl_wikitext(model, encoded)
print(f"Overall Perplexity for the dataset: {overall_perplexity}")

# Note: The activation['mlp.gate_proj'] will be populated after the model has processed some input.
# If you want to inspect activations, ensure this line is placed after model inference or evaluation.
# print(activation['mlp.gate_proj'])
