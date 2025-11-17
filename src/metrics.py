import numpy as np
from collections import Counter
from tokens import itos, tokens_to_notes


# ---------------------------
# 1. N-GRAM REPETITION
# ---------------------------
def ngram_repetition(tokens, n=4):
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


# ---------------------------
# 2. SILENCE RATIO
# ---------------------------
def silence_ratio(notes):
    if len(notes) == 0:
        return 1.0
    
    silence = 0.0
    last_end = 0.0
    
    for (_, start, end, _) in notes:
        if start > last_end:
            silence += (start - last_end)
        last_end = end
    
    return silence / last_end if last_end > 0 else 1.0


# ---------------------------
# 3. NOTE DENSITY
# ---------------------------
def note_density(notes):
    if len(notes) < 2:
        return 0.0
    total_time = notes[-1][2] - notes[0][1]
    return len(notes) / total_time if total_time > 0 else 0.0


# ---------------------------
# 4. PITCH RANGE
# ---------------------------
def pitch_range(notes):
    if len(notes) == 0:
        return (0, 0)
    pitches = [n[0] for n in notes]
    return min(pitches), max(pitches)


# ---------------------------
# 5. INTERVAL STATISTICS
# ---------------------------
def interval_stats(notes):
    if len(notes) < 2:
        return (0, 0)
    pitches = [n[0] for n in notes]
    intervals = [abs(pitches[i+1]-pitches[i]) for i in range(len(pitches)-1)]
    return np.mean(intervals), np.max(intervals)


# ---------------------------
# 6. DURATION CONSISTENCY
# ---------------------------
def duration_stats(notes):
    if len(notes) == 0:
        return (0,0)
    durations = [(end-start) for (_,start,end,_) in notes]
    return np.mean(durations), np.std(durations)


# ---------------------------
# 7. MAIN METRIC FUNCTION
# ---------------------------
def compute_all_metrics(tokens):
    """
    Input: token ids
    Output: dict of metrics
    """
    # Convert tokens → notes
    notes = tokens_to_notes(tokens)

    metrics = {}

    # Basic
    metrics["num_tokens"] = len(tokens)
    metrics["num_notes"] = len(notes)

    # Duration
    if len(notes) > 0:
        metrics["duration_sec"] = notes[-1][2]
    else:
        metrics["duration_sec"] = 0.0

    # Core metrics
    metrics["note_density"] = note_density(notes)
    metrics["silence_ratio"] = silence_ratio(notes)
    metrics["repetition_4gram"] = ngram_repetition(tokens, 4)

    # Intermediate metrics
    lo, hi = pitch_range(notes)
    metrics["pitch_min"] = lo
    metrics["pitch_max"] = hi

    mean_int, max_int = interval_stats(notes)
    metrics["mean_interval"] = mean_int
    metrics["max_interval"] = max_int

    mean_dur, std_dur = duration_stats(notes)
    metrics["mean_duration"] = mean_dur
    metrics["std_duration"] = std_dur

    return metrics


# ---------------------------
# 8. PRETTY PRINT
# ---------------------------
def print_metrics(metrics, title="Metrics"):
    print("\n====================")
    print(title)
    print("====================")
    for k, v in metrics.items():
        print(f"{k:20s} : {v}")
    print("====================\n")