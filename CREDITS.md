# Credits

sparkyard stands on a lot of other people's work. Thank you.

## Inspiration

sparkyard began as an exploration of **mARTin-B78's DGX Spark stack**
(<https://github.com/mARTin-B78/dgx-spark_lite-llm_llama-swap_vllm_llama-cpp_ollama>),
which introduced the idea of running LiteLLM + llama-swap + vLLM + llama.cpp +
Ollama together on the NVIDIA DGX Spark. sparkyard is an **independent
reimplementation** built around a single-source-of-truth generator; it shares
no code with that project and is not a fork.

## Tooling & image builds

- **[@eugr](https://github.com/eugr)** — [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  (MIT), which builds the SM121 `vllm-node` images sparkyard serves vLLM models
  with, and [llama-benchy](https://github.com/eugr/llama-benchy), the throughput
  benchmark `make bench MODE=speed` wraps.
- **[@christopherowen](https://github.com/christopherowen)** —
  spark-vllm-mxfp4-docker, the CUTLASS MXFP4 build for GPT-OSS-class models.
- **[@SeraphimSerapis](https://github.com/SeraphimSerapis)** —
  [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench), the
  tool-calling quality benchmark `make bench` wraps.

## Upstream projects

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [llama-swap](https://github.com/mostlygeek/llama-swap)
- [LiteLLM](https://github.com/BerriAI/litellm)
- [vLLM](https://github.com/vllm-project/vllm)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [Ollama](https://github.com/ollama/ollama)
- [Open WebUI](https://github.com/open-webui/open-webui)
