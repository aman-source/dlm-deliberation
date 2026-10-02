"""In-canvas scratch, pre-read thought, and anytime slot readouts.

Greedy and deterministic: temperature 0, no Gumbel noise. Ties in confidence
break toward the earlier scratch position.

NFE is T + 1 for C1 and C2 (T unmasking passes, then one read with the
scratch filled) and 1 for C0. The answer slot is never written.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from dd.labels import slot_distribution

LogitsFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def kt_schedule(scratch_length: int, steps: int) -> list[int]:
    """How many scratch tokens to unmask at each step.

    Earlier steps take the remainder so the counts differ by at most one and
    sum to S. Requires T <= S.
    """
    if steps <= 0:
        raise ValueError("T must be positive")
    if scratch_length <= 0:
        raise ValueError("S must be positive")
    if steps > scratch_length:
        raise ValueError(f"T ({steps}) must be <= S ({scratch_length})")
    base = scratch_length // steps
    remainder = scratch_length % steps
    return [base + (1 if index < remainder else 0) for index in range(steps)]


def nfe_for(condition: str, steps: int | None) -> int:
    if condition == "C0":
        return 1
    if condition in {"C1", "C2", "C3"}:
        if steps is None:
            raise ValueError(f"{condition} requires T")
        return int(steps) + 1
    raise ValueError(f"unknown condition {condition}")


def _pad(batch_ids: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(ids) for ids in batch_ids)
    batch = torch.full((len(batch_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch_ids), width), dtype=torch.long)
    for row, ids in enumerate(batch_ids):
        batch[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        mask[row, : len(ids)] = 1
    return batch, mask


def _confidence_and_token(
    logits_row: torch.Tensor,
    mask_id: int | None = None,
    suppressed_ids: Sequence[int] | None = None,
) -> tuple[float, int]:
    """Confidence and argmax token for one scratch position.

    `suppressed_ids` get -inf logits before the softmax (suppress_eos_in_scratch), so both the
    chosen token and the confidence used to rank positions are over the remaining vocabulary.
    """
    logits_row = logits_row.float()
    if suppressed_ids:
        logits_row = logits_row.clone()
        logits_row[torch.as_tensor(list(suppressed_ids), dtype=torch.long, device=logits_row.device)] = float("-inf")
    probs = torch.softmax(logits_row, dim=0)
    # Never "fill" a scratch position with the mask token itself; it would stay masked
    # and break the k_t accounting. Confidence is still the full-vocab softmax.
    choice = probs.clone()
    if mask_id is not None and 0 <= mask_id < choice.numel():
        choice[mask_id] = -1.0
    token = int(torch.argmax(choice).item())
    return float(probs[token].item()), token


def end_fraction(scratch_ids: Sequence[int], end_ids: Sequence[int]) -> float | None:
    """Share of filled scratch tokens that are any end-type id (eos, pad, end-of-turn, ...)."""
    if not scratch_ids:
        return None
    ends = {int(i) for i in end_ids}
    return sum(1 for token in scratch_ids if int(token) in ends) / len(scratch_ids)


def _select_positions(masked: Sequence[int], confidence: Sequence[float], k: int) -> list[int]:
    order = sorted(range(len(masked)), key=lambda index: (-confidence[index], masked[index]))
    return order[:k]


def run_c0(
    logits_fn: LogitsFn,
    canvases: list,
    letter_to_ids: dict[str, list[int]],
    letters_per_item: list[list[str]],
    mask_id: int,
    pad_id: int,
) -> list[dict]:
    del mask_id  # the slot is already masked in the canvas
    batch, attn = _pad([canvas.input_ids for canvas in canvases], pad_id)
    logits = logits_fn(batch, attn)
    rows = []
    for index, canvas in enumerate(canvases):
        read = slot_distribution(logits[index, canvas.slot_index], letter_to_ids, letters_per_item[index])
        rows.append(
            {
                "readouts": [
                    {
                        "step": 1,
                        "probs": read["probs"],
                        "letter_logits": read["letter_logits"],
                        "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                    }
                ],
                "final_probs": read["probs"],
                "letter_logits": read["letter_logits"],
                "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                "pred": read["pred"],
                "nfe": 1,
                "scratch_text": "",
                "filled_ids": canvas.input_ids,
            }
        )
    return rows


def run_c1(
    logits_fn: LogitsFn,
    canvases: list,
    letter_to_ids: dict[str, list[int]],
    letters_per_item: list[list[str]],
    mask_id: int,
    pad_id: int,
    steps: int,
    tokenizer=None,
    end_ids: Sequence[int] = (),
    suppress_end_in_scratch: bool = False,
) -> list[dict]:
    scratch_length = len(canvases[0].scratch_positions)
    for canvas in canvases:
        if len(canvas.scratch_positions) != scratch_length:
            raise ValueError("a C1 batch must share S")
    schedule = kt_schedule(scratch_length, steps)
    sequences = [list(canvas.input_ids) for canvas in canvases]
    suppressed = list(end_ids) if suppress_end_in_scratch else None
    readouts: list[list[dict]] = [[] for _ in canvases]

    for step, k in enumerate(schedule, start=1):
        batch, attn = _pad(sequences, pad_id)
        logits = logits_fn(batch, attn)
        for index, canvas in enumerate(canvases):
            read = slot_distribution(
                logits[index, canvas.slot_index], letter_to_ids, letters_per_item[index]
            )
            readouts[index].append(
                {
                    "step": step,
                    "probs": read["probs"],
                    "letter_logits": read["letter_logits"],
                    "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                }
            )
            masked = [pos for pos in canvas.scratch_positions if sequences[index][pos] == mask_id]
            if len(masked) < k:
                raise RuntimeError(f"step {step} asked to unmask {k} but only {len(masked)} remain")
            confidence = []
            tokens = []
            for pos in masked:
                conf, token = _confidence_and_token(logits[index, pos], mask_id, suppressed)
                confidence.append(conf)
                tokens.append(token)
            chosen = _select_positions(masked, confidence, k)
            for choice in chosen:
                sequences[index][masked[choice]] = tokens[choice]
            if sequences[index][canvas.slot_index] != mask_id:
                raise RuntimeError("answer slot was unmasked during scratch denoising")

    batch, attn = _pad(sequences, pad_id)
    logits = logits_fn(batch, attn)
    rows = []
    for index, canvas in enumerate(canvases):
        if any(sequences[index][pos] == mask_id for pos in canvas.scratch_positions):
            raise RuntimeError("scratch was not fully filled before the final read")
        read = slot_distribution(logits[index, canvas.slot_index], letter_to_ids, letters_per_item[index])
        readouts[index].append(
            {
                "step": steps + 1,
                "probs": read["probs"],
                "letter_logits": read["letter_logits"],
                "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                "final": True,
            }
        )
        scratch_ids = [sequences[index][pos] for pos in canvas.scratch_positions]
        scratch_text = ""
        if tokenizer is not None:
            scratch_text = tokenizer.decode(scratch_ids, skip_special_tokens=False)
        rows.append(
            {
                "readouts": readouts[index],
                "final_probs": read["probs"],
                "letter_logits": read["letter_logits"],
                "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                "pred": read["pred"],
                "nfe": steps + 1,
                "scratch_text": scratch_text,
                "eos_pad_fraction": end_fraction(scratch_ids, end_ids),
                "suppress_eos_in_scratch": bool(suppress_end_in_scratch),
                "filled_ids": sequences[index],
            }
        )
    return rows


def run_c2(
    logits_fn: LogitsFn,
    canvases: list,
    letter_to_ids: dict[str, list[int]],
    letters_per_item: list[list[str]],
    mask_id: int,
    pad_id: int,
    steps: int,
    tokenizer,
    template: str,
    end_ids: Sequence[int] = (),
    suppress_end_in_scratch: bool = False,
) -> list[dict]:
    """Denoise a thought canvas with no answer slot, then append the slot and read once."""
    from dd.canvas import append_answer_slot

    scratch_length = len(canvases[0].scratch_positions)
    schedule = kt_schedule(scratch_length, steps)
    sequences = [list(canvas.input_ids) for canvas in canvases]
    suppressed = list(end_ids) if suppress_end_in_scratch else None
    for step, k in enumerate(schedule, start=1):
        del step
        batch, attn = _pad(sequences, pad_id)
        logits = logits_fn(batch, attn)
        for index, canvas in enumerate(canvases):
            masked = [pos for pos in canvas.scratch_positions if sequences[index][pos] == mask_id]
            if len(masked) < k:
                raise RuntimeError(f"thought step asked to unmask {k} but only {len(masked)} remain")
            confidence = []
            tokens = []
            for pos in masked:
                conf, token = _confidence_and_token(logits[index, pos], mask_id, suppressed)
                confidence.append(conf)
                tokens.append(token)
            chosen = _select_positions(masked, confidence, k)
            for choice in chosen:
                sequences[index][masked[choice]] = tokens[choice]

    answered = []
    for index, canvas in enumerate(canvases):
        filled = type(canvas)(
            input_ids=sequences[index],
            slot_index=None,
            scratch_positions=list(canvas.scratch_positions),
            prompt_len=canvas.prompt_len,
            condition="C2",
            template=template,
        )
        answered.append(append_answer_slot(tokenizer, filled, mask_id, template=template))
    batch, attn = _pad([canvas.input_ids for canvas in answered], pad_id)
    logits = logits_fn(batch, attn)
    rows = []
    for index, canvas in enumerate(answered):
        read = slot_distribution(logits[index, canvas.slot_index], letter_to_ids, letters_per_item[index])
        scratch_ids = [canvas.input_ids[pos] for pos in canvas.scratch_positions]
        rows.append(
            {
                "readouts": [
                    {
                        "step": steps + 1,
                        "probs": read["probs"],
                        "letter_logits": read["letter_logits"],
                        "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                        "final": True,
                    }
                ],
                "final_probs": read["probs"],
                "letter_logits": read["letter_logits"],
                "off_label_mass": read["off_label_mass"],
                        "slot_topk": read["slot_topk"],
                "pred": read["pred"],
                "nfe": steps + 1,
                "scratch_text": tokenizer.decode(scratch_ids, skip_special_tokens=False),
                "eos_pad_fraction": end_fraction(scratch_ids, end_ids),
                "suppress_eos_in_scratch": bool(suppress_end_in_scratch),
                "filled_ids": canvas.input_ids,
            }
        )
    return rows


def flip_rate(rows: list[dict]) -> float:
    """Mean, over consecutive recorded readouts, of the fraction of items that change argmax.

    C1 records steps 1..T and the final pass as step T+1. C0 and C2 have a single
    readout, so the rate is undefined and this returns NaN.
    """
    if not rows:
        return float("nan")
    n_steps = len(rows[0].get("readouts") or [])
    if n_steps < 2:
        return float("nan")
    rates = []
    for step in range(n_steps - 1):
        changed = 0
        for row in rows:
            left = _argmax_letter(row["readouts"][step]["probs"])
            right = _argmax_letter(row["readouts"][step + 1]["probs"])
            changed += int(left != right)
        rates.append(changed / len(rows))
    return float(sum(rates) / len(rates))


def _argmax_letter(probs: dict[str, float]) -> str:
    letters = list(probs)
    return max(letters, key=lambda letter: (probs[letter], -letters.index(letter)))
