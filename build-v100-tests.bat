@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.EXE" --build C:\Users\Josh\Documents\GitHub\Strata\build-cuda12 -j 10 --target native_expert_parity mmvq_multi_parity gr_parity gdn_parity qsa_parity kv_q8_parity elementwise_parity sampler_parity router_top10_parity rope_parity quantize_act_parity shared_expert_parity bf16_gemv_parity s_gemv_q8k_parity strata-device
