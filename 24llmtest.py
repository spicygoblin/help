# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibivibiv/alpaca-dragon-72b-v1")
model = AutoModelForCausalLM.from_pretrained("ibivibiv/alpaca-dragon-72b-v1")

inputs = tokenizer("### Instruction: Create a plan for developing the game of snake in python using pygame.\n### Response:\n",
                   return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
