# tokens.py
# Generic symbolic event tokens for monophonic music:
# NOTE_ON_p (0..127), NOTE_OFF_p (0..127), TIME_SHIFT_s (1..8 16th-note steps), VEL_{L,M,H}, BAR, EOS

import numpy as np

PITCHES = list(range(0,128))
NOTE_ON = [f"NOTE_ON_{p}" for p in PITCHES]
NOTE_OFF = [f"NOTE_OFF_{p}" for p in PITCHES]
TSHIFT = [f"TIME_SHIFT_{i}" for i in range(1,9)]  # 1..8 sixteenth-note steps
VELS = ["VEL_L", "VEL_M", "VEL_H"]
SPECIAL = ["BAR", "EOS"]

VOCAB = NOTE_ON + NOTE_OFF + TSHIFT + VELS + SPECIAL
stoi = {t:i for i,t in enumerate(VOCAB)}
itos = {i:t for t,i in stoi.items()}

def note_to_events(pitch, dur_steps, vel):
    """Returns [NOTE_ON_p, VEL_bin, TIME_SHIFT_dur, NOTE_OFF_p]."""
    vbin = "VEL_L" if vel < 43 else ("VEL_M" if vel < 86 else "VEL_H")
    dur_steps = max(1, min(8, int(dur_steps)))
    return [f"NOTE_ON_{pitch}", vbin, f"TIME_SHIFT_{dur_steps}", f"NOTE_OFF_{pitch}"]

def notes_to_tokens(notes):
    """notes: list of (pitch 0..127, dur_steps 1..8, vel 0..127)"""
    toks = []
    steps_in_bar = 16
    acc = 0
    for (p, d, v) in notes:
        events = note_to_events(int(p), int(d), int(v))
        toks.extend(events)
        acc += d
        if acc >= steps_in_bar:
            toks.append("BAR")
            acc -= steps_in_bar
    toks.append("EOS")
    return [stoi[t] for t in toks]

def tokens_to_notes(tokens, default_vel=90, step=0.125):
    """
    Convert event tokens to a list of notes:
    (pitch, start, end, velocity)
    """
    time = 0.0
    notes = []
    active_notes = {}  # pitch -> start time

    for tok in tokens:
        t = itos[tok]

        # 1. TIME SHIFT
        if t.startswith("TIME_SHIFT_"):
            amount = int(t.split("_")[2])
            time += amount * step   # move time forward

        # 2. NOTE ON
        elif t.startswith("NOTE_ON_"):
            pitch = int(t.split("_")[2])
            active_notes[pitch] = time  # start note

        # 3. NOTE OFF
        elif t.startswith("NOTE_OFF_"):
            pitch = int(t.split("_")[2])
            if pitch in active_notes:
                start = active_notes[pitch]
                end = time if time > start else start + step
                notes.append((pitch, start, end, default_vel))
                del active_notes[pitch]

        # 4. End of sequence
        elif t == "EOS":
            break

    # Close any open notes at end
    for pitch, start in active_notes.items():
        end = time + step
        notes.append((pitch, start, end, default_vel))

    return notes


# ---- Tiny demo dataset (C-major-ish toy phrases) ----
DEMO_MELODIES = [
    # (pitch, dur_steps(1..8), vel)
    [(60,2,80),(62,2,80),(64,2,85),(65,2,90),(67,4,88),(65,2,84),(64,2,80),(62,4,78)],
    [(60,2,78),(64,2,82),(67,4,86),(69,2,88),(67,2,84),(65,2,82),(64,2,80),(60,4,78)],
    [(62,2,80),(64,2,80),(65,2,82),(67,2,84),(69,2,86),(67,2,84),(65,2,82),(64,4,80)],
    [(67,2,85),(65,2,82),(64,2,80),(62,2,78),(60,4,76),(62,2,78),(64,2,80),(60,4,76)],
]

def load_demo_tokens():
    return [notes_to_tokens(m) for m in DEMO_MELODIES]
