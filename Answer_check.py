from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
import random
import os

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

def quantize_weights(layer, v):
    with torch.no_grad():
        quantized_weights = torch.round(layer.weight * v) 
        quantized_weights = torch.clamp(quantized_weights, -8, 7)
        quantized_weights = quantized_weights / v
        layer.weight.copy_(quantized_weights)


def quantize_all_linear_layers(model, v):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = 7/((torch.mean(torch.abs(module.weight))+1e-8)) * v
            quantize_weights(module, w)

def zero_weights(layer):
    with torch.no_grad():
        layer.weight.fill_(0)

def generate_answer(model, tokenizer, question):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def test_questions(model, tokenizer, questions):
    print("Testing model:")
    for question in questions:
        answer = generate_answer(model, tokenizer, question)
        print(f"Q: {answer}\n")

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-chat-hf"
set_seed(1)

# Define your questions
questions = [
    "Who wrote Lord of the Rings?",
    "Calculate 2+3",
    "What is the capital of France?",
    "Who is the president of the United States?",
    "What is the largest mammal?",
    "What is 'self' according to Carl Jung?",
    "What is the meaning of life?",
    "What is the best programming language?",
    "What is the best movie of all time?",
    "What is the best book ever written?"
]

# Initialize and test the original model
model, tokenizer = initialize_model_and_tokenizer(model_name)
model.to(device)
test_questions(model, tokenizer, questions)

# Quantization levels and zero weight test
vs = [2 ** i for i in range(5)]

del model  # Delete the model
torch.cuda.empty_cache()

for v in vs:
    # Re-initialize model and quantize weights
    model, _ = initialize_model_and_tokenizer(model_name)
    quantize_all_linear_layers(model, v)
    model.to(device)
    # layer = model.model.layers[-1].mlp.gate_proj  # Adjust the path to your layer
    # quantize_weights(layer, v)
    
    # Test after quantization
    print(f"After quantization with level v={v}:")
    test_questions(model, tokenizer, questions)
    del model  # Delete the model
    torch.cuda.empty_cache()

