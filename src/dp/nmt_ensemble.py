"""Token-level (logits) averaging ensemble for encoder-decoder NMT models.

This module intentionally keeps dependencies light and avoids relying on
`model.generate()` so we can combine multiple independently trained checkpoints.

Supported (minimal) decoding features:
  - greedy (num_beams=1)
  - beam search (num_beams>=2)
  - repetition_penalty
  - no_repeat_ngram_size
  - length_penalty (beam finalization)

Notes:
  - Designed for HuggingFace Seq2Seq models (e.g., T5/ByT5).
  - For speed we use `past_key_values` caching per model.
  - Ensemble is done at *token-level*: logits are averaged before softmax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import math

import torch


@dataclass
class EnsembleGenConfig:
    max_new_tokens: int
    num_beams: int = 1
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0


def _apply_repetition_penalty_(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    repetition_penalty: float,
) -> None:
    """In-place repetition penalty (HF-compatible behavior).

    Mirrors `transformers.generation.logits_process.RepetitionPenaltyLogitsProcessor`.
    """

    if repetition_penalty is None or repetition_penalty == 1.0:
        return

    # Gather scores for all tokens that have been generated so far.
    # logits: (batch, vocab), sequences: (batch, cur_len)
    gathered = logits.gather(1, sequences)
    gathered = torch.where(gathered < 0, gathered * repetition_penalty, gathered / repetition_penalty)
    logits.scatter_(1, sequences, gathered)


def _apply_no_repeat_ngram_(
    logits: torch.Tensor,
    sequences: torch.Tensor,
    no_repeat_ngram_size: int,
) -> None:
    """In-place no-repeat ngram constraint (GPU-friendly).

    NOTE:
      The common HF implementation uses `input_ids.tolist()` internally which forces a
      GPU->CPU sync + copy every decoding step. That makes generation (and especially
      ensembles) *much* slower and can look "CPU-speed" even on a GPU.

      This implementation stays on the current device using tensor ops (unfold + compare).
      It bans tokens that would create a repeated n-gram of size `no_repeat_ngram_size`.
    """

    n = int(no_repeat_ngram_size)
    if n <= 0:
        return

    bsz, cur_len = sequences.size()
    # We are selecting the NEXT token; need at least n-1 previous tokens to form an n-gram.
    if cur_len + 1 < n:
        return

    n1 = n - 1

    # windows: (bsz, cur_len - n1 + 1, n1) == (bsz, cur_len - n + 2, n-1)
    windows = sequences.unfold(1, n1, 1)

    # Compare all windows except the last (it has no next-token to ban).
    prev_windows = windows[:, :-1, :]  # (bsz, cur_len - n + 1, n-1)
    last = sequences[:, -n1:]          # (bsz, n-1)

    # matches: (bsz, cur_len - n + 1) — start positions whose (n-1)-gram matches the tail
    matches = (prev_windows == last[:, None, :]).all(dim=-1)

    # next_tokens[j] is the token that followed the (n-1)-gram at start position j
    next_tokens = sequences[:, n1:]    # (bsz, cur_len - n + 1)

    # Ban those next tokens
    idx_i, idx_j = torch.where(matches)
    banned = next_tokens[idx_i, idx_j]
    logits[idx_i, banned] = -float("inf")

def _avg_logits(logits_list: Sequence[torch.Tensor]) -> torch.Tensor:
    """Average logits across models in float32 for stability."""

    if len(logits_list) == 1:
        return logits_list[0]
    acc = logits_list[0].float()
    for t in logits_list[1:]:
        acc = acc + t.float()
    return acc / float(len(logits_list))


def _length_penalize(score: float, length: int, length_penalty: float) -> float:
    if length_penalty is None or length_penalty == 1.0:
        return score
    # Common length penalty form: score / (length ** lp)
    # (Constant +1 from decoder_start is shared across all beams; negligible.)
    denom = float(max(1, length)) ** float(length_penalty)
    return score / denom


@torch.inference_mode()
def ensemble_generate(
    models: List[torch.nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_cfg: EnsembleGenConfig,
) -> torch.Tensor:
    """Generate sequences by averaging logits across `models`.

    Returns:
        Tensor[int64] of shape (batch, seq_len) with left-aligned sequences.
        Sequences are padded with `pad_token_id` to the max length in the batch.
    """

    if not models:
        raise ValueError("models must be a non-empty list")

    device = input_ids.device
    batch_size = input_ids.size(0)
    num_beams = int(gen_cfg.num_beams or 1)
    max_new_tokens = int(gen_cfg.max_new_tokens)

    pad_id = int(gen_cfg.pad_token_id)
    eos_id = int(gen_cfg.eos_token_id)
    start_id = int(gen_cfg.decoder_start_token_id)

    # Greedy
    if num_beams <= 1:
        # Encoder outputs per model (no beam expansion)
        enc_outs: List[Tuple[torch.Tensor]] = []
        for m in models:
            enc = m.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            enc_outs.append((enc.last_hidden_state,))

        sequences = torch.full((batch_size, 1), start_id, dtype=torch.long, device=device)
        past_list: List[Optional[object]] = [None for _ in models]
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        for _step in range(max_new_tokens):
            step_input = sequences if past_list[0] is None else sequences[:, -1:]
            step_logits: List[torch.Tensor] = []
            for mi, m in enumerate(models):
                out = m(
                    encoder_outputs=enc_outs[mi],
                    attention_mask=attention_mask,
                    decoder_input_ids=step_input,
                    past_key_values=past_list[mi],
                    use_cache=True,
                    return_dict=True,
                )
                step_logits.append(out.logits[:, -1, :])
                past_list[mi] = out.past_key_values

            logits = _avg_logits(step_logits)
            _apply_repetition_penalty_(logits, sequences, gen_cfg.repetition_penalty)
            _apply_no_repeat_ngram_(logits, sequences, gen_cfg.no_repeat_ngram_size)

            next_tokens = torch.argmax(logits, dim=-1)
            # Force pad after eos for already finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

            sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
            finished = finished | (next_tokens == eos_id)
            if bool(torch.all(finished)):
                break

        return sequences

    # Beam search (minimal)
    # Expand inputs for beams
    input_ids_beam = input_ids.repeat_interleave(num_beams, dim=0)
    attn_beam = attention_mask.repeat_interleave(num_beams, dim=0)

    # Encoder outputs per model (compute once on original batch, then expand)
    enc_outs_beam: List[Tuple[torch.Tensor]] = []
    for m in models:
        enc = m.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        enc_hidden = enc.last_hidden_state.repeat_interleave(num_beams, dim=0)
        enc_outs_beam.append((enc_hidden,))

    # Sequences and scores
    sequences = torch.full((batch_size * num_beams, 1), start_id, dtype=torch.long, device=device)
    beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
    beam_scores[:, 0] = 0.0
    beam_scores = beam_scores.view(-1)  # (batch*num_beams,)

    past_list: List[Optional[object]] = [None for _ in models]
    finalized: List[List[Tuple[float, List[int]]]] = [[] for _ in range(batch_size)]
    done = torch.zeros((batch_size,), dtype=torch.bool, device=device)

    vocab_size: Optional[int] = None

    for _step in range(max_new_tokens):
        # Forward each model
        step_input = sequences if past_list[0] is None else sequences[:, -1:]

        step_logits: List[torch.Tensor] = []
        for mi, m in enumerate(models):
            out = m(
                encoder_outputs=enc_outs_beam[mi],
                attention_mask=attn_beam,
                decoder_input_ids=step_input,
                past_key_values=past_list[mi],
                use_cache=True,
                return_dict=True,
            )
            step_logits.append(out.logits[:, -1, :])
            past_list[mi] = out.past_key_values

        logits = _avg_logits(step_logits)
        if vocab_size is None:
            vocab_size = logits.size(-1)

        # Apply logits processors
        _apply_repetition_penalty_(logits, sequences, gen_cfg.repetition_penalty)
        _apply_no_repeat_ngram_(logits, sequences, gen_cfg.no_repeat_ngram_size)

        # Convert to log-probs
        log_probs = torch.log_softmax(logits, dim=-1)

        # Add previous beam scores
        next_scores = log_probs + beam_scores.unsqueeze(-1)  # (batch*num_beams, vocab)

        # Reshape to (batch, num_beams*vocab)
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)

        # For each example, select candidates
        topk = min(num_beams * 2, num_beams * vocab_size)
        topk_scores, topk_indices = torch.topk(next_scores, k=topk, dim=-1)

        next_beam_scores: List[float] = []
        next_beam_tokens: List[int] = []
        next_beam_indices: List[int] = []

        # CPU-side loop is OK: batch is small (inference batch_size)
        for b in range(batch_size):
            if bool(done[b]):
                # Keep dummy beams (pad), to keep shapes consistent.
                for bi in range(num_beams):
                    next_beam_scores.append(float(-1e9))
                    next_beam_tokens.append(pad_id)
                    next_beam_indices.append(b * num_beams + bi)
                continue

            beams_for_b: List[Tuple[float, int, int]] = []  # (score, prev_global_beam, token)

            for score, idx in zip(topk_scores[b].tolist(), topk_indices[b].tolist()):
                beam_id = idx // vocab_size
                token_id = idx % vocab_size
                global_beam = b * num_beams + beam_id

                if token_id == eos_id:
                    # Finalize: store sequence including EOS
                    seq = sequences[global_beam].tolist() + [eos_id]
                    finalized[b].append((float(score), seq))
                    continue

                beams_for_b.append((float(score), global_beam, int(token_id)))
                if len(beams_for_b) >= num_beams:
                    break

            if len(beams_for_b) < num_beams:
                # Not enough active beams => mark done.
                done[b] = True
                # Fill remaining with dummy pads
                while len(beams_for_b) < num_beams:
                    beams_for_b.append((-1e9, b * num_beams, pad_id))

            for score, prev_beam, tok in beams_for_b:
                next_beam_scores.append(score)
                next_beam_tokens.append(tok)
                next_beam_indices.append(prev_beam)

        beam_scores = torch.tensor(next_beam_scores, dtype=torch.float, device=device)
        beam_tokens = torch.tensor(next_beam_tokens, dtype=torch.long, device=device)
        beam_indices = torch.tensor(next_beam_indices, dtype=torch.long, device=device)

        # Reorder sequences
        sequences = sequences.index_select(0, beam_indices)
        sequences = torch.cat([sequences, beam_tokens.unsqueeze(1)], dim=1)

        # Reorder caches per model (if available)
        for mi, m in enumerate(models):
            if past_list[mi] is None:
                continue
            # Prefer model's reorder helper (handles new cache types too).
            if hasattr(m, "_reorder_cache"):
                past_list[mi] = m._reorder_cache(past_list[mi], beam_indices)
            else:
                # Fallback: best-effort index_select on nested tuples.
                past_list[mi] = _index_select_nested(past_list[mi], beam_indices)

        # If all are done, we can stop early.
        if bool(torch.all(done)):
            break

    # Select best hypothesis per example
    best_seqs: List[List[int]] = []
    seq_len = sequences.size(1)
    for b in range(batch_size):
        cands: List[Tuple[float, List[int]]] = []
        for score, seq in finalized[b]:
            cands.append((_length_penalize(score, len(seq), gen_cfg.length_penalty), seq))

        # Also consider current active beams (not EOS)
        for bi in range(num_beams):
            global_beam = b * num_beams + bi
            score = float(beam_scores[global_beam].item())
            seq = sequences[global_beam].tolist()
            cands.append((_length_penalize(score, len(seq), gen_cfg.length_penalty), seq))

        # Pick best
        cands.sort(key=lambda x: x[0], reverse=True)
        best_seqs.append(cands[0][1] if cands else [start_id])

    # Pad to max length
    max_len = max(len(s) for s in best_seqs) if best_seqs else 1
    out = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(best_seqs):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return out


def _index_select_nested(obj: object, idx: torch.Tensor) -> object:
    """Fallback cache reordering when model._reorder_cache is unavailable."""

    if obj is None:
        return None
    if torch.is_tensor(obj):
        return obj.index_select(0, idx)
    if isinstance(obj, tuple):
        return tuple(_index_select_nested(x, idx) for x in obj)
    if isinstance(obj, list):
        return [_index_select_nested(x, idx) for x in obj]
    # Unknown type: return as-is
    return obj
