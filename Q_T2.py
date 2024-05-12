import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load a pre-trained model and tokenizer
pretrained_model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

# Convert to AutoGPTQ model with quantization config
quantize_config = BaseQuantizeConfig(bits=4)
quantized_model = AutoGPTQForCausalLM.from_pretrained(
    pretrained_model_name,
    quantize_config=quantize_config,
    torch_dtype=torch.float16
)

# Ensure CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare example data for quantization
example_texts = [
    "Write a bubble search in C++: #include <iostream>\n"
    "using namespace std;\n"
    "\n"
    "void bubbleSort(int array[], int size) {\n"
    "    for (int step = 0; step < size - 1; ++step) {\n"
    "        bool swapped = false;\n"
    "        for (int i = 0; i < size - step - 1; ++i) {\n"
    "            if (array[i] > array[i + 1]) {\n"
    "                // Swap if the element found is greater than the next element\n"
    "                int temp = array[i];\n"
    "                array[i] = array[i + 1];\n"
    "                array[i + 1] = temp;\n"
    "                swapped = true;\n"
    "            }\n"
    "        }\n"
    "        // If no two elements were swapped by inner loop, then break\n"
    "        if (!swapped) {\n"
    "            break;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "\n"
    "void printArray(int array[], int size) {\n"
    "    for (int i = 0; i < size; i++) {\n"
    "        cout << array[i] << ' ';\n"
    "    }\n"
    "    cout << endl;\n"
    "}\n"
    "\n"
    "int main() {\n"
    "    int data[] = {-2, 45, 0, 11, -9};\n"
    "    int size = sizeof(data) / sizeof(data[0]);\n"
    "    \n"
    "    bubbleSort(data, size);\n"
    "    \n"
    "    cout << \"Sorted Array in Ascending Order:\\n\";\n"
    "    printArray(data, size);\n"
    "}\n"
]


# Tokenize the example texts and ensure they are in the correct structure and on the correct device
examples = [tokenizer.encode_plus(text, return_tensors="pt", return_attention_mask=True) for text in example_texts]
examples = [{key: tensor.to(device, dtype=torch.float16) for key, tensor in example.items()} for example in examples]

# Quantize the model with the examples
quantized_model.quantize(examples)

# Generate an answer to the prompt
task_prompt = "Write a bubble sort in C++"
inputs = tokenizer(task_prompt, return_tensors="pt").to(device, dtype=torch.float16)
outputs = quantized_model.generate(**inputs, max_length=500)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)
