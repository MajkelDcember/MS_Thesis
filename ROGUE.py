from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
import random
import os
from rouge_score import rouge_scorer

def compute_rouge(model, tokenizer, questions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for question, reference in zip(questions, references):
        generated_answer = generate_answer(model, tokenizer, question)
        score = scorer.score(reference, generated_answer)
        scores.append(score)
    return scores

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def find_max_infinity_norm(model):
    max_inf_norm = 0
    for param in model.parameters():
        max_inf_norm = max(max_inf_norm, param.data.abs().max().item())
    return max_inf_norm

def quantize_weights(layer, v):
    with torch.no_grad():
        quantized_weights = torch.round(layer.weight * v) 
        quantized_weights = torch.clamp(quantized_weights, -8, 7)
        quantized_weights = quantized_weights/ v
        layer.weight.copy_(quantized_weights)

def quantize_all_linear_layers(model, v):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            quantize_weights(module, v)

def generate_answer(model, tokenizer, question):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-chat-hf"
set_seed(42)

# Load dataset and prepare questions and references
dataset = load_dataset("squad", split='validation')
questions = [item['question'] for item in dataset]
references = [item['answers']['text'][0] for item in dataset]

# Initialize model and tokenizer
model, tokenizer = initialize_model_and_tokenizer(model_name)
model.to(device)

# Calculate the maximum infinity norm
max_infinity_norm = find_max_infinity_norm(model)

# Define quantization levels
K_values = [2 ** i for i in range(5)]
vs = [(32767 / max_infinity_norm) * K for K in K_values]

# Compute ROUGE scores before quantization
rouge_scores = compute_rouge(model, tokenizer, questions, references)
print("ROUGE scores:", rouge_scores)

# Quantization example (optional)
for v in vs:
    model, _ = initialize_model_and_tokenizer(model_name)  # Re-initialize model
    quantize_all_linear_layers(model, v)  # Apply quantization
    model.to(device)
    print(f"After quantization with level v={v}:")
    rouge_scores = compute_rouge(model, tokenizer, questions, references)
    print("ROUGE scores:", rouge_scores)
    del model  # Ensure to delete model after each iteration
    torch.cuda.empty_cache()  # Clear GPU cache

# Clean up
del model
torch.cuda.empty_cache()
