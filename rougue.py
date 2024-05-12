from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import numpy as np
import random
import os
from rouge_score import rouge_scorer
dataset = load_dataset("squad")



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

def generate_answer(model, tokenizer, question):
    input_ids = tokenizer.encode(question, return_tensors='pt').to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_rouge(model, tokenizer, questions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = []
    for question, reference in zip(questions, references):
        generated_answer = generate_answer(model, tokenizer, question)
        score = scorer.score(reference, generated_answer)
        scores.append(score)
    return scores

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-2-7b-chat-hf"
set_seed(42)

# Define your questions and references
questions = [item['question'] for item in dataset['validation']]
references = [item['answers']['text'][0] for item in dataset['validation']]  # using the first answer as the reference


# Initialize and test the original model
model, tokenizer = initialize_model_and_tokenizer(model_name)
model.to(device)

# Evaluate using ROUGE
rouge_scores = compute_rouge(model, tokenizer, questions, references)
print("ROUGE scores:", rouge_scores)

del model  # Delete the model
torch.cuda.empty_cache()

# Continue with the rest of your code for quantization and further evaluations...
