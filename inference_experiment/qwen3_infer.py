from transformers import AutoModelForCausalLM, AutoTokenizer
import time

# model_name = "Qwen/Qwen3-30B-A3B"
model_name = "Qwen/Qwen3-32B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# Print the chosen dtype and device map
print(f"Model loaded with dtype: {model.config.torch_dtype}")
print(f"Model device map: {model.hf_device_map}")

# prepare the model input
prompt = "How up to date are you regarding CVEs?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switch between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
start_time = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)
print(f"Inference time: {time.time()-start_time:.2f} seconds")