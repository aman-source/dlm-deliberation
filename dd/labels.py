"""Letter-slot probability from single-token variants.

For each letter L, variants are the single-token encodings of "L" and " L".
The slot score for L is the logsumexp of those variants' logits. Slot
probabilities are a softmax across the item's letters.

off_label_mass is 1 minus the full-vocabulary softmax mass on every variant
of every valid letter.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch


class LabelResolutionError(RuntimeError):
    pass


# ARC-Challenge has one 5-option test item (arc-TIMSS_2007_8_pg53), so E must resolve too.
ALL_LETTERS: tuple[str, ...] = ("A", "B", "C", "D", "E")
SLOT_TOPK = 5


def resolve_letter_tokens(tokenizer, letters: Sequence[str] = ALL_LETTERS) -> dict[str, list[int]]:
    """Collect single-token variants. Abort if any letter has none."""
    resolved: dict[str, list[int]] = {}
    missing: list[str] = []
    for letter in letters:
        ids: list[int] = []
        for variant in (letter, f" {letter}"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                token_id = int(encoded[0])
                if token_id not in ids:
                    ids.append(token_id)
        if not ids:
            missing.append(letter)
        else:
            resolved[letter] = ids
    if missing:
        raise LabelResolutionError(
            "No single-token variant for letter(s): "
            + ", ".join(missing)
            + ". Tried 'L' and ' L' with add_special_tokens=False."
        )
    return resolved


# Substrings that mark an end-type special token (end of text, end of turn, eos, pad).
# "<|end_header_id|>" is a role delimiter, not an end token, and matches none of these.
END_TOKEN_MARKERS = ("endoftext", "end_of_text", "eot", "eos", "end_of_turn", "im_end", "</s>", "pad")


def resolve_end_tokens(tokenizer, extra_ids: Sequence[int | None] = ()) -> dict[int, str]:
    """Every end-type special token id -> its string.

    Sources: tokenizer eos/pad ids, any special or added token whose text contains an
    END_TOKEN_MARKERS substring, and `extra_ids` (for example model.config eos/pad ids).
    """
    found: dict[int, str] = {}

    def add(token_id) -> None:
        if token_id is None:
            return
        ids = token_id if isinstance(token_id, (list, tuple)) else [token_id]
        for value in ids:
            if value is not None and int(value) >= 0:
                found[int(value)] = tokenizer.convert_ids_to_tokens(int(value))

    add(getattr(tokenizer, "eos_token_id", None))
    add(getattr(tokenizer, "pad_token_id", None))
    for token_id in extra_ids:
        add(token_id)
    candidates = dict(getattr(tokenizer, "added_tokens_encoder", {}) or {})
    for token in getattr(tokenizer, "all_special_tokens", []) or []:
        candidates.setdefault(token, tokenizer.convert_tokens_to_ids(token))
    for token, token_id in candidates.items():
        lowered = token.lower() if isinstance(token, str) else ""
        # Qwen-family vocabularies (Dream) carry multimodal/FIM placeholders such as <|image_pad|>;
        # they are not end-of-sequence tokens.
        if any(skip in lowered for skip in ("fim_", "image", "video", "vision")):
            continue
        if lowered and any(marker in lowered for marker in END_TOKEN_MARKERS):
            add(token_id)
    if not found:
        raise LabelResolutionError("no end-type special tokens found in the tokenizer")
    return dict(sorted(found.items()))


def verify_mask_token(tokenizer, expected_id: int = 126336, expected_token: str = "<|mdm_mask|>") -> dict:
    """VERIFY the LLaDA mask id. Returns what the tokenizer actually says."""
    observed = tokenizer.convert_ids_to_tokens(expected_id)
    special = dict(getattr(tokenizer, "special_tokens_map", {}) or {})
    added = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    special_hits = []
    for key, value in special.items():
        if "mask" in str(key).lower() or (isinstance(value, str) and "mask" in value.lower()):
            special_hits.append({"key": key, "value": value})
    added_hits = []
    if isinstance(added, dict):
        for token, token_id in added.items():
            if isinstance(token, str) and "mask" in token.lower():
                added_hits.append({"token": token, "id": int(token_id)})
    # [gMASK] also contains the substring "mask". Prefer the expected string,
    # then an added token whose text equals it, and do not treat gMASK as the
    # diffusion mask when <|mdm_mask|> is present.
    adopted = None
    if observed != expected_token:
        for hit in added_hits:
            if hit["token"] == expected_token:
                adopted = hit
                break
        if adopted is None:
            for hit in added_hits:
                if "mdm_mask" in hit["token"] or hit["token"] == "<|mask|>":
                    adopted = hit
                    break
    return {
        "expected_id": expected_id,
        "expected_token": expected_token,
        "convert_ids_to_tokens": observed,
        "matches_expected_string": observed == expected_token,
        "special_tokens_map_mask": special_hits,
        "added_tokens_containing_mask": added_hits,
        "adopted_because_expected_mismatched": adopted,
    }


def slot_distribution(
    logits: torch.Tensor,
    letter_to_ids: Mapping[str, Sequence[int]],
    letters: Sequence[str],
) -> dict:
    """Read one position's logits into a letter distribution.

    `logits` is a 1-D vocab vector. Scores are logsumexp over variant logits,
    then softmax across `letters` only.
    """
    if logits.ndim != 1:
        raise ValueError(f"expected a vocab vector, got shape {tuple(logits.shape)}")
    scores = []
    for letter in letters:
        ids = list(letter_to_ids[letter])
        if not ids:
            raise LabelResolutionError(f"letter {letter} has no token ids")
        index = torch.tensor(ids, device=logits.device, dtype=torch.long)
        scores.append(torch.logsumexp(logits.index_select(0, index).float(), dim=0))
    score_vec = torch.stack(scores)
    probs = torch.softmax(score_vec, dim=0)

    all_ids = sorted({int(i) for letter in letters for i in letter_to_ids[letter]})
    log_probs = torch.log_softmax(logits.float(), dim=0)
    index = torch.tensor(all_ids, device=logits.device, dtype=torch.long)
    labeled_mass = torch.exp(torch.logsumexp(log_probs.index_select(0, index), dim=0))
    off_label_mass = float((1.0 - labeled_mass).item())

    # Diagnostic: which full-vocab tokens actually hold the slot's mass (explains off_label_mass).
    top_p, top_i = torch.topk(torch.exp(log_probs), k=min(SLOT_TOPK, log_probs.numel()))
    slot_topk = [[int(i), float(p)] for p, i in zip(top_p.tolist(), top_i.tolist())]

    prob_map = {letter: float(probs[i].item()) for i, letter in enumerate(letters)}
    logit_map = {letter: float(score_vec[i].item()) for i, letter in enumerate(letters)}
    pred = max(prob_map, key=lambda letter: (prob_map[letter], -letters.index(letter)))
    return {
        "probs": prob_map,
        "letter_logits": logit_map,
        "off_label_mass": off_label_mass,
        "slot_topk": slot_topk,
        "pred": pred,
    }
