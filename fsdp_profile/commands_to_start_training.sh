# On one node
NCCL_DEBUG_PROTO=1 FORCE_UNIFORM_ROUTING=1 MASTER_ADDR=28.59.81.136 MASTER_PORT=7865 RANK=0 bash ./fsdp_profile/run_and_profile_moe.sh ./fsdp_profile/Mistral_8_7B

# One another node
NCCL_DEBUG_PROTO=1 FORCE_UNIFORM_ROUTING=1 MASTER_ADDR=28.49.38.169 MASTER_PORT=7865 RANK=1 bash ./fsdp_profile/run_and_profile_moe.sh ./fsdp_profile/Mistral_8_7B