export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_MODELSCOPE=False

# python -m vllm.entrypoints.openai.api_server \
# --model /share/shared_models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/92011f62d7604e261f748ec0cfe6329f31193e33 \
# --served-model-name llama2 \
# --tensor-parallel-size 4 \
# --trust-remote-code \

# python -m vllm.entrypoints.openai.api_server \
# --model /share/kelong/rag_studio/baselines/ft_llama2_zero_shot/merged_checkpoint \
# --served-model-name ft_llama2 \
# --tensor-parallel-size 4 \
# --trust-remote-code \


# python -m vllm.entrypoints.openai.api_server \
# --model /share/kelong/rag_studio/baselines/raft_bge_llama2_three_shot_rag/merged_checkpoint \
# --served-model-name raft_llama2 \
# --tensor-parallel-size 4 \
# --trust-remote-code \


python -m vllm.entrypoints.openai.api_server \
--model /share/kelong/rag_studio/baselines/raft_bge_llama2_three_shot_rag/chat_merged_checkpoint \
--served-model-name chat_raft_llama2 \
--tensor-parallel-size 4 \
--trust-remote-code \


# python -m vllm.entrypoints.openai.api_server \
# --model Qwen/Qwen1.5-7B-Chat \
# --served-model-name qwen \
# --tensor-parallel-size 4 \
# --trust-remote-code \


# python -m vllm.entrypoints.openai.api_server \
# --model /share/kelong/rag_studio/baselines/tmp/merged_checkpoint \
# --served-model-name tmp \
# --tensor-parallel-size 4 \
# --trust-remote-code \