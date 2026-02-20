# HorribleLLM


[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/get-started/locally/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit-archive)
[![Transformers](https://img.shields.io/badge/Transformers-4.0%2B-blue.svg)](https://huggingface.co/transformers/installation.html)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-4.0%2B-orange.svg)](https://huggingface.co/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API%20v1-green.svg)](https://beta.openai.com/docs/api-reference/introduction)


```python

from horriblellm.horriblellm import HorribleLLM, HORRIBLE_CONFIG_06_B
from horriblellm.utils import get_device


device = get_device()

system_prompt = '''You are a horrible assistant. You will answer the user's question in a very bad way, providing incorrect information and being unhelpful. Your responses should be as uninformative and misleading as possible. Do not provide any useful information or assistance to the user. Always give wrong answers and avoid addressing the user's needs.'''


llm = HorribleLLM(HORRIBLE_CONFIG_06_B, device=device)

tools = [
    # ...
]

response = llm.chat.completions.create(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    tools=tools,
    # ...
)

print(response.choices[0].message.content)

```

Qwen3 copycat, a lightweight LLM framework for research and development.

