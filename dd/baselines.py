"""Autoregressive letter readout.

Qwen3-8B is the System One baseline: chat template with enable_thinking=False,
then the same pinned answer prefix used for C0, then the next-token distribution
over letter variants. NFE is 1. No scratch canvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dd.labels import ALL_LETTERS, resolve_letter_tokens, slot_distribution
from dd.llada import NanLogits


@dataclass
class CausalModel:
    name: str
    hf_id: str
    tokenizer: object
    model: object
    device: torch.device
    dtype: torch.dtype
    pad_id: int
    revision: str | None
    letter_tokens: dict[str, list[int]]
    thinking_flag: str

    @torch.inference_mode()  # no autograd graph: without this, long canvases OOM a 24GB GPU
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = output.logits
        if not torch.isfinite(logits).all():
            raise NanLogits("NaN or Inf in causal logits")
        return logits


def load_causal_model(
    hf_id: str, name: str, device: torch.device, dtype: torch.dtype, revision: str | None = None
) -> tuple[CausalModel, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, revision=revision)
    thinking_flag = "enable_thinking=False"
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        thinking_flag = "enable_thinking unsupported; template called without it"
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        revision=revision,
        low_cpu_mem_usage=True,
        device_map="cuda" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model.to(device)
    model.eval()
    letters = resolve_letter_tokens(tokenizer, ALL_LETTERS)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    revision = getattr(model.config, "_commit_hash", None) or revision
    bundle = CausalModel(
        name=name,
        hf_id=hf_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
        pad_id=int(pad_id),
        revision=revision,
        letter_tokens=letters,
        thinking_flag=thinking_flag,
    )
    return bundle, thinking_flag


def render_causal_prompt(tokenizer, prompt: str) -> str:
    kwargs = {"add_generation_prompt": True, "tokenize": False}
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            enable_thinking=False,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def read_causal_batch(
    bundle: CausalModel,
    prompts: list[str],
    letters_per_item: list[list[str]],
    answer_prefix: str,
) -> list[dict]:
    """Sequence ends with the answer prefix. Logits at that last token are the letter distribution."""
    sequences = []
    for prompt in prompts:
        text = render_causal_prompt(bundle.tokenizer, prompt)
        prefix = list(bundle.tokenizer.encode(text, add_special_tokens=False))
        answer = list(bundle.tokenizer.encode(answer_prefix, add_special_tokens=False))
        if not answer:
            raise ValueError(f"answer prefix {answer_prefix!r} encoded to zero tokens")
        sequences.append(prefix + answer)
    width = max(len(ids) for ids in sequences)
    batch = torch.full((len(sequences), width), bundle.pad_id, dtype=torch.long)
    attn = torch.zeros((len(sequences), width), dtype=torch.long)
    last_index = []
    for row, ids in enumerate(sequences):
        batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attn[row, : len(ids)] = 1
        last_index.append(len(ids) - 1)
    logits = bundle.forward(batch, attn)
    rows = []
    for index, letters in enumerate(letters_per_item):
        read = slot_distribution(logits[index, last_index[index]], bundle.letter_tokens, letters)
        rows.append(
            {
                "readouts": [
                    {
                        "step": 1,
                        "probs": read["probs"],
                        "letter_logits": read["letter_logits"],
                        "off_label_mass": read["off_label_mass"],
                    }
                ],
                "final_probs": read["probs"],
                "letter_logits": read["letter_logits"],
                "off_label_mass": read["off_label_mass"],
                "pred": read["pred"],
                "nfe": 1,
                "scratch_text": "",
            }
        )
    return rows
