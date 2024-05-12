from transformers import AutoTokenizer, TextGenerationPipeline, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
import torch
import gc 



Prompt = "Here is a funny joke: "
max_tokenz = 100

pretrained_model_dir = "mistralai/Mistral-7B-v0.1"
quantized_model_dir_4bit = "mistralai/Mistral-7B-v0.1-4bit"
quantized_model_dir_2bit = "mistralai/Mistral-7B-v0.1-2bit"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
model_4bit = AutoGPTQForCausalLM.from_quantized(quantized_model_dir_4bit, device="cuda:0")

pipeline = TextGenerationPipeline(model=model_4bit, tokenizer=tokenizer)
quantized_output_4bit = pipeline(Prompt, max_new_tokens=max_tokenz)[0]["generated_text"]
print("Quantized model 4-bit output:")
print(quantized_output_4bit)


model_2bit = AutoGPTQForCausalLM.from_quantized(quantized_model_dir_2bit, device="cuda:0")

pipeline = TextGenerationPipeline(model=model_2bit, tokenizer=tokenizer)
quantized_output_2bit = pipeline(Prompt, max_new_tokens=max_tokenz)[0]["generated_text"]
print("Quantized model 2-bit output:")
print(quantized_output_2bit)




unquantized_model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
unquantized_model.to(device)
unquantized_text = tokenizer.decode(unquantized_model.generate(**tokenizer(Prompt, return_tensors="pt").to(device), max_new_tokens=max_tokenz)[0])
print("Unquantized model output:")
print(unquantized_text)
print("Quantized model 4-bit output:")
print(quantized_output_4bit)
print("Quantized model 2-bit output:")
print(quantized_output_2bit)