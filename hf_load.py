from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFacePipeline  

# -------------------- Load Model --------------------
model_name = ".............."

# load model from device
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512, 
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id)

llm = HuggingFacePipeline(pipeline=pipe)

