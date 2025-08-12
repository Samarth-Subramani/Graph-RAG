from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline  

# -------------------- Load Model --------------------
local_model_path = "/home/vault/iwia/iwia125h/models/deepseek-8b"

# load model from device
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512, 
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id)

llm = HuggingFacePipeline(pipeline=pipe)

