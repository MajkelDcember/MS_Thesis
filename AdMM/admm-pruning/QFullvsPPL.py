import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
import matplotlib.pyplot as plt

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers, prune_admm
from lib.eval import eval_ppl, eval_zero_shot

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )
    model.seqlen = model.config.max_position_embeddings 
    return model

def quantize_weights(layer, v):
    with torch.no_grad():
        quantized_weights = torch.round(layer.weight * v)
        quantized_weights = torch.clamp(quantized_weights, -8, 7)
        quantized_weights = quantized_weights / v
        layer.weight.copy_(quantized_weights)

def quantize_all_linear_layers(model, v):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            w = 32767/(128*128*(module.weight.abs().mean()+1e-8)) * v
            quantize_weights(module, w)

def find_max_infinity_norm(model):
    max_inf_norm = 0
    for param in model.parameters():
        max_inf_norm = max(max_inf_norm, param.data.abs().max().item())
    return max_inf_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="/scratch/llm_weights", type=str)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = get_llm(args.model, args.cache_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    
    vs = [2 ** i for i in range(15)]
    perplexities = []
    unquantized_ppl = eval_ppl(args.model,model, tokenizer, device)
    print(f"unquantized model Perplexity: {unquantized_ppl}")
    for v in vs:
        model = get_llm(args.model, args.cache_dir).to(device)
        quantize_all_linear_layers(model, v)
        ppl = eval_ppl(args.model,model, tokenizer, device)
        perplexities.append(ppl)
        print(f"Quantization v={v}, Perplexity: {ppl}")
        del model
        torch.cuda.empty_cache()

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

if __name__ == '__main__':
    main()
