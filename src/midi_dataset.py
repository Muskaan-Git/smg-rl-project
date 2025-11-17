import os
import pretty_midi
from tokens import notes_to_tokens

def midi_to_note_events(path, quant_step=0.125):
    """
    Convert a monophonic MIDI file into (pitch, dur_steps, velocity) tuples.
    quant_step = length of 1 time-step in seconds or beats.
    """
    pm = pretty_midi.PrettyMIDI(path)
    if len(pm.instruments) == 0:
        return []

    # Use first instrument (usually main melody for simple datasets)
    inst = pm.instruments[0]
    notes = sorted(inst.notes, key=lambda n: n.start)

    events = []
    for note in notes:
        pitch = note.pitch
        vel = note.velocity
        duration = note.end - note.start

        # Convert duration into integer number of TIME_SHIFT steps
        dur_steps = max(1, int(round(duration / quant_step)))

        events.append((pitch, dur_steps, vel))

    return events


def load_midi_folder(folder):
    """
    Loads all .mid / .midi files in a folder and converts them to token sequences.
    Returns: list of sequences, where each sequence is a list of token IDs.
    """
    seqs = []
    for name in os.listdir(folder):
        if not name.lower().endswith((".mid", ".midi")):
            continue

        path = os.path.join(folder, name)
        try:
            events = midi_to_note_events(path)
            if len(events) == 0:
                continue

            tokens = notes_to_tokens(events)
            seqs.append(tokens)
        except Exception as e:
            print("Skipping:", path, "Error:", e)

    return seqs
