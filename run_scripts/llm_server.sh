export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_MODELSCOPE=False

python -m vllm.entrypoints.openai.api_server \
--model /share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33 \
--served-model-name llama2 \
--tensor-parallel-size 4 \
--trust-remote-code \