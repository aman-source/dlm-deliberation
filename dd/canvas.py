"""Token canvases for C0, C1, and C2.

The user prompt is rendered with the tokenizer chat template, then canvas
tokens are appended. Pinned tokens are never masked or overwritten. The answer
slot is a single mask and is never written by the denoiser.

Protocol v1.1 templates (`v1e`, `v3e`) pin `<|eot_id|>` right after the slot, so
the slot is no longer the terminal position, and give C1/C2 a pinned
"Let me think step by step.\\n" before the scratch:

  C0: prompt + c0_prefix + [slot] + <|eot_id|>
  C1: prompt + think_prefix + [S masks] + scratch_prefix + [slot] + <|eot_id|>
  C2: prompt + think_prefix + [S masks]            (denoised; no template or slot visible)
      then + scratch_prefix + [slot] + <|eot_id|>  (read once)

`v1`/`v2`/`v3` are the protocol v1.0 smoke templates, kept so old rows stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

THINK_PREFIX = "Let me think step by step.\n"
EOT = "<|eot_id|>"

TEMPLATES: dict[str, dict] = {
    "v1": {
        "c0_prefix": "Answer:",
        "scratch_prefix": "\nAnswer:",
        "note": "v1.0 plan default. Pinned 'Answer:' for C0 and '\\nAnswer:' after scratch.",
    },
    "v2": {
        "c0_prefix": "Answer: ",
        "scratch_prefix": "\nAnswer: ",
        "note": "v1.0. Trailing space, so the slot can land on a leading-space letter token.",
    },
    "v3": {
        "c0_prefix": "The answer is",
        "scratch_prefix": "\nThe answer is",
        "note": "v1.0. Alternate wording tried when Gate 1 rejected v1 on off-label mass.",
    },
    "v1e": {
        "c0_prefix": "Answer:",
        "scratch_prefix": "\nAnswer:",
        "think_prefix": THINK_PREFIX,
        "slot_suffix": [EOT],
        "note": "Protocol v1.1: v1 wording, think prefix, <|eot_id|> pinned after the slot.",
    },
    "v3e_dream": {
        "c0_prefix": "The answer is",
        "scratch_prefix": "\nThe answer is",
        "think_prefix": THINK_PREFIX,
        "slot_suffix": ["<|im_end|>"],
        "note": "v1.3 Dream: v3e with Dream/Qwen's end-of-turn token pinned after the slot.",
    },
    "v1e_dream": {
        "c0_prefix": "Answer:",
        "scratch_prefix": "\nAnswer:",
        "think_prefix": THINK_PREFIX,
        "slot_suffix": ["<|im_end|>"],
        "note": "v1.3 Dream: v1e with Dream/Qwen's end-of-turn token pinned after the slot.",
    },
    "v3e": {
        "c0_prefix": "The answer is",
        "scratch_prefix": "\nThe answer is",
        "think_prefix": THINK_PREFIX,
        "slot_suffix": [EOT],
        "note": "Protocol v1.1 default: v3 wording, think prefix, <|eot_id|> pinned after the slot.",
    },
}


@dataclass
class Canvas:
    input_ids: list[int]
    slot_index: int | None
    scratch_positions: list[int] = field(default_factory=list)
    prompt_len: int = 0
    condition: str = "C0"
    template: str = "v1"


def encode_prompt(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def encode_prefix(tokenizer, text: str) -> list[int]:
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError(f"answer prefix {text!r} encoded to zero tokens")
    return ids


def encode_suffix(tokenizer, spec: dict) -> list[int]:
    """Special tokens pinned after the slot, by exact token string (never split by the BPE)."""
    ids = []
    for token in spec.get("slot_suffix", []):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or tokenizer.convert_ids_to_tokens(int(token_id)) != token:
            raise ValueError(f"slot suffix token {token!r} is not a single vocabulary token")
        ids.append(int(token_id))
    return ids


def _think(tokenizer, spec: dict) -> list[int]:
    text = spec.get("think_prefix")
    return encode_prefix(tokenizer, text) if text else []


def build_canvas(
    tokenizer,
    prompt: str,
    condition: str,
    scratch_length: int,
    mask_id: int,
    template: str = "v1",
    prompt_ids: list[int] | None = None,
) -> Canvas:
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template}")
    spec = TEMPLATES[template]
    prefix = list(prompt_ids) if prompt_ids is not None else encode_prompt(tokenizer, prompt)
    suffix = encode_suffix(tokenizer, spec)
    if condition == "C0":
        answer = encode_prefix(tokenizer, spec["c0_prefix"])
        slot = len(prefix) + len(answer)
        return Canvas(
            input_ids=prefix + answer + [mask_id] + suffix,
            slot_index=slot,
            scratch_positions=[],
            prompt_len=len(prefix),
            condition="C0",
            template=template,
        )
    if condition in ("C1", "C2"):
        if scratch_length <= 0:
            raise ValueError(f"{condition} requires S > 0")
        think = _think(tokenizer, spec)
        start = len(prefix) + len(think)
        scratch = list(range(start, start + scratch_length))
        ids = prefix + think + [mask_id] * scratch_length
        if condition == "C2":
            return Canvas(
                input_ids=ids,
                slot_index=None,
                scratch_positions=scratch,
                prompt_len=len(prefix),
                condition="C2",
                template=template,
            )
        answer = encode_prefix(tokenizer, spec["scratch_prefix"])
        slot = len(ids) + len(answer)
        return Canvas(
            input_ids=ids + answer + [mask_id] + suffix,
            slot_index=slot,
            scratch_positions=scratch,
            prompt_len=len(prefix),
            condition="C1",
            template=template,
        )
    raise ValueError(f"unknown condition {condition}")


def append_answer_slot(tokenizer, canvas: Canvas, mask_id: int, template: str = "v1") -> Canvas:
    """After C2 thought tokens are filled, pin the answer prefix, one mask, and the slot suffix."""
    spec = TEMPLATES[template]
    answer = encode_prefix(tokenizer, spec["scratch_prefix"])
    slot = len(canvas.input_ids) + len(answer)
    return Canvas(
        input_ids=list(canvas.input_ids) + answer + [mask_id] + encode_suffix(tokenizer, spec),
        slot_index=slot,
        scratch_positions=list(canvas.scratch_positions),
        prompt_len=canvas.prompt_len,
        condition="C2",
        template=template,
    )
