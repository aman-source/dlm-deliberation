"""LLaDA (and other masked dLLM) loading, mask-token check, and forward.

Weights are not downloaded until `load_masked_model` is called. Tokenizer-only
verification is enough to check the mask id and the letter variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dd.labels import ALL_LETTERS, resolve_end_tokens, resolve_letter_tokens, verify_mask_token


class AttentionMaskRejected(RuntimeError):
    """The loaded forward does not accept attention_mask. Caller must use batch size 1."""


class NanLogits(RuntimeError):
    """A forward pass produced NaN or Inf logits."""


@dataclass
class MaskedModel:
    name: str
    hf_id: str
    tokenizer: object
    model: object
    device: torch.device
    dtype: torch.dtype
    mask_id: int
    pad_id: int
    revision: str | None
    letter_tokens: dict[str, list[int]]
    end_tokens: dict[int, str] = field(default_factory=dict)
    attention_mask_ok: bool = True
    # Dream (adapted from an AR model) predicts the token at position i from position i-1; its own
    # generation code shifts logits right by one. LLaDA does not.
    logit_shift: bool = False
    accepts_use_cache: bool = True

    @torch.inference_mode()  # no autograd graph: without this, long canvases OOM a 24GB GPU
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        use_mask = self.attention_mask_ok and bool((attention_mask == 0).any().item())
        kwargs = {"input_ids": input_ids}
        if self.accepts_use_cache:
            kwargs["use_cache"] = False
        if use_mask:
            kwargs["attention_mask"] = attention_mask
        try:
            output = self.model(**kwargs)
        except TypeError as exc:
            if use_mask and "attention_mask" in str(exc):
                self.attention_mask_ok = False
                raise AttentionMaskRejected(str(exc)) from exc
            if "use_cache" in str(exc) and self.accepts_use_cache:
                self.accepts_use_cache = False
                kwargs.pop("use_cache")
                output = self.model(**kwargs)
            else:
                raise
        logits = output.logits
        if self.logit_shift:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        if not torch.isfinite(logits).all():
            raise NanLogits("NaN or Inf in logits")
        return logits


def load_tokenizer(hf_id: str, revision: str | None = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, revision=revision)


def repo_revision(hf_id: str) -> str | None:
    try:
        from huggingface_hub import model_info

        return model_info(hf_id).sha
    except Exception:
        return None


def load_masked_model(
    hf_id: str,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    mask_id: int = 126336,
    expected_mask_token: str = "<|mdm_mask|>",
    revision: str | None = None,
    logit_shift: bool = False,
) -> tuple[MaskedModel, dict]:
    tokenizer = load_tokenizer(hf_id, revision)
    mask_report = verify_mask_token(tokenizer, mask_id, expected_mask_token)
    if not mask_report["matches_expected_string"]:
        replacement = mask_report.get("adopted_because_expected_mismatched")
        if replacement and replacement.get("id") is not None:
            mask_id = int(replacement["id"])
            mask_report["adopted_id"] = mask_id
            mask_report["adopted_token"] = replacement.get("token")
        else:
            raise RuntimeError(
                f"mask id {mask_id} is {mask_report['convert_ids_to_tokens']!r}, "
                f"not {expected_mask_token!r}, and no added mask token was found"
            )
    letters = resolve_letter_tokens(tokenizer, ALL_LETTERS)
    from transformers import AutoModel

    # Weights go straight to the GPU; no full CPU copy first.
    model = AutoModel.from_pretrained(
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
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    bundle = MaskedModel(
        name=name,
        hf_id=hf_id,
        tokenizer=tokenizer,
        model=model,
        device=device,
        dtype=dtype,
        mask_id=int(mask_id),
        pad_id=int(pad_id),
        revision=getattr(model.config, "_commit_hash", None) or revision or repo_revision(hf_id),
        letter_tokens=letters,
        logit_shift=logit_shift,
        end_tokens=resolve_end_tokens(
            tokenizer,
            extra_ids=(getattr(model.config, "eos_token_id", None), getattr(model.config, "pad_token_id", None)),
        ),
    )
    return bundle, mask_report


def probe_dtype(load_fn, dtypes: list[torch.dtype]) -> tuple[MaskedModel, dict, torch.dtype]:
    """Try dtypes in order. Keep the first that loads and returns finite logits.

    `load_fn(dtype) -> (MaskedModel, mask_report)`. A dtype that raises on load
    or on a 32-token probe is discarded. One dtype is then frozen.
    """
    errors = []
    for dtype in dtypes:
        try:
            bundle, mask_report = load_fn(dtype)
            ids = torch.full((1, 32), bundle.pad_id, dtype=torch.long)
            ids[0, -1] = bundle.mask_id
            attn = torch.ones(1, 32, dtype=torch.long)
            bundle.forward(ids, attn)
            return bundle, mask_report, dtype
        except Exception as exc:
            errors.append(f"{dtype}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no dtype produced finite logits: " + " | ".join(errors))
