import math

import torch
import torch.nn.functional as F

from .tokenizer import BOS, EOS, PAD, encode, decode


def _length_score(logp, generated_length, alpha):
    # BOS is excluded; an emitted EOS contributes one token to the length.
    return logp / (((5 + generated_length) / 6) ** alpha)


def _next_logps(model, memory, src_pad, sequences):
    # A model may avoid projecting every old target position during inference.
    forward = getattr(model, "decode_step_logits", model.decode_logits)
    logits = forward(memory, src_pad, sequences)[:, -1]
    logps = F.log_softmax(logits.float(), dim=-1)
    logps[:, PAD] = -float("inf")
    logps[:, BOS] = -float("inf")
    return logps


def _greedy(model, memory, src_pad, max_tokens):
    device = memory.device
    owners = list(range(memory.size(0)))
    sequences = torch.full((len(owners), 1), BOS, dtype=torch.long, device=device)
    results = [None] * len(owners)
    for step in range(max_tokens):
        tokens = _next_logps(model, memory, src_pad, sequences).argmax(dim=-1)
        sequences = torch.cat((sequences, tokens[:, None]), dim=1)
        token_list = tokens.tolist()
        keep = []
        for index, (owner, token) in enumerate(zip(owners, token_list)):
            if token == EOS or step + 1 == max_tokens:
                results[owner] = sequences[index].tolist()
            else:
                keep.append(index)
        if not keep:
            break
        indices = torch.tensor(keep, dtype=torch.long, device=device)
        sequences = sequences.index_select(0, indices)
        memory = memory.index_select(0, indices)
        src_pad = src_pad.index_select(0, indices)
        owners = [owners[index] for index in keep]
    return results


def _beam(model, memory, src_pad, beam_size, alpha, max_tokens):
    device = memory.device
    count = memory.size(0)
    # All live prefixes have equal length. A single decoder call covers every
    # sentence and beam; finished sentences disappear from subsequent calls.
    active = [(owner, [BOS], 0.0) for owner in range(count)]
    best_done = [None] * count
    results = [None] * count
    for step in range(max_tokens):
        owners = torch.tensor([owner for owner, _, _ in active], device=device)
        sequences = torch.tensor([seq for _, seq, _ in active], device=device)
        logps = _next_logps(model, memory.index_select(0, owners),
                           src_pad.index_select(0, owners), sequences)
        # Keep each parent's top 2K candidates, INCLUDING EOS, before merging.
        # Their union contains the sentence's global top 2K: a child ranked
        # below 2K within its own parent cannot enter the global top 2K.
        # This reduces the CPU transfer without changing global beam pruning.
        width = min(2 * beam_size, logps.size(-1))
        values, tokens = torch.topk(logps, width, dim=-1)
        values, tokens = values.tolist(), tokens.tolist()
        candidates = {}
        for index, (owner, seq, score) in enumerate(active):
            bucket = candidates.setdefault(owner, [])
            for value, token in zip(values[index], tokens[index]):
                if math.isfinite(value):
                    bucket.append((seq + [token], score + value))
        active = []
        for owner, bucket in candidates.items():
            bucket.sort(key=lambda candidate: candidate[1], reverse=True)
            next_active = []
            for rank, (seq, score) in enumerate(bucket[:2 * beam_size]):
                if seq[-1] == EOS:
                    # Standard beam completion: only EOS among the global
                    # top K may finish. A lower-ranked EOS is pruned rather
                    # than resurrected later because its short score is high.
                    if rank < beam_size:
                        normalized = _length_score(score, step + 1, alpha)
                        if best_done[owner] is None or normalized > best_done[owner][2]:
                            best_done[owner] = (seq, score, normalized)
                elif len(next_active) < beam_size:
                    # The extra K candidates refill live slots occupied by EOS.
                    next_active.append((seq, score))
            bucket = next_active
            done = best_done[owner]
            if step + 1 == max_tokens or not bucket:
                pool = bucket + ([(done[0], done[1])] if done else [])
                if not pool:
                    raise ValueError("decoder has no finite legal token probability")
                results[owner] = max(pool, key=lambda candidate:
                    _length_score(candidate[1], len(candidate[0]) - 1, alpha))[0]
            elif done is not None and alpha >= 0 and done[2] >= max(
                    _length_score(score, max_tokens, alpha) for _, score in bucket):
                # Future log probabilities cannot exceed zero. For alpha >= 0,
                # the maximum permitted length gives an optimistic bound.
                results[owner] = done[0]
            else:
                active.extend((owner, seq, score) for seq, score in bucket)
        if not active:
            break
    return results


@torch.inference_mode()
def generate_batch_ids(model, sources, method="greedy", beam_size=4, alpha=0.6, max_tokens=256):
    """Return token sequences and flags for output reaching max_tokens.

    Source sequences are padded in full. A limit flag is true even when EOS is
    emitted on the final permitted step; `sequence[-1] != EOS` identifies an
    unfinished result separately. Length includes EOS and excludes initial BOS.
    """
    if (method not in ("greedy", "beam") or beam_size < 1 or max_tokens < 1
            or not math.isfinite(alpha)):
        raise ValueError("invalid decoding parameters")
    if not sources:
        return [], []
    if any(not source for source in sources):
        raise ValueError("source token sequences must be nonempty")
    if method == "beam" and beam_size == 1:
        method = "greedy"
    device = next(model.parameters()).device
    src = torch.full((len(sources), max(map(len, sources))), PAD,
                     dtype=torch.long, device=device)
    for index, ids in enumerate(sources):
        src[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        memory, src_pad = model.encode(src)
        sequences = (_greedy(model, memory, src_pad, max_tokens) if method == "greedy"
                     else _beam(model, memory, src_pad, beam_size, alpha, max_tokens))
        return sequences, [len(sequence) - 1 >= max_tokens for sequence in sequences]
    finally:
        model.train(was_training)


def generate_ids(model, src_ids, method="greedy", beam_size=4, alpha=0.6, max_tokens=256):
    sequences, hits = generate_batch_ids(model, [src_ids], method, beam_size, alpha, max_tokens)
    return sequences[0], hits[0]


def translate_batch(model, tok, texts, method="greedy", beam_size=4, alpha=0.6, max_tokens=256):
    sequences, hits = generate_batch_ids(model, [encode(tok, text) for text in texts],
                                         method, beam_size, alpha, max_tokens)
    return [decode(tok, sequence) for sequence in sequences], hits


def translate(model, tok, text, method="greedy", beam_size=4, alpha=0.6, max_tokens=256):
    predictions, hits = translate_batch(model, tok, [text], method, beam_size, alpha, max_tokens)
    return predictions[0], hits[0]
