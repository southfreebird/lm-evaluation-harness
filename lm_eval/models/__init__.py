from . import gpt2
from . import gpt3
from . import anthropic_llms
# from . import huggingface  # Remove gpu depandencies: peft 
from . import textsynth
from . import dummy

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    # Remove gpu depandencies: peft
    # "hf-causal-experimental": huggingface.AutoCausalLM,
    # "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "anthropic": anthropic_llms.AnthropicLM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
