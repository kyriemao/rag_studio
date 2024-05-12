from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from IPython import embed

tokenizer = AutoTokenizer.from_pretrained("/share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33")
chat = [ {"role": "user", "content": "Hello, how are you?"},
]
x = tokenizer.apply_chat_template(chat, tokenize=False)

embed()
input()

