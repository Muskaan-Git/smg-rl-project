import argparse
import torch
from torch.distributions.categorical import Categorical
import pretty_midi
import soundfile as sf

from model import TinyTransformerLM
from tokens import VOCAB, itos, tokens_to_notes


# ---------- SAMPLING FUNCTION ----------

def sample(model, T=256, temperature=1.0):
    """Autoregressively sample a sequence of token IDs from the model.

    Args:
        model: TinyTransformerLM instance.
        T: Maximum number of sampling steps.
        temperature: Softmax temperature for sampling.

    Returns:
        List[int]: generated token ids.
    """
    model.eval()
    x = torch.zeros(1, 1, dtype=torch.long)  # start with a dummy token 0
    tokens = []

    # If the model exposes a max_len attribute, use it to limit context
    max_len = getattr(model, "max_len", None)

    for _ in range(T):
        # If sequence is longer than model's max_len, keep only the last window
        if max_len is not None and x.size(1) > max_len:
            x = x[:, -max_len:]

        # Safety: ensure all ids are in range before calling the model
        assert x.max().item() < len(VOCAB), (
            f"Found id {x.max().item()} >= vocab {len(VOCAB)} in x"
        )

        # Forward pass and sample next token
        logits = model(x)[:, -1, :]          # [1, vocab]
        logits = logits / temperature
        dist = Categorical(logits=logits)
        token_id = int(dist.sample().item())

        # Safety clamp: avoid out-of-range token IDs
        if token_id < 0 or token_id >= len(VOCAB):
            token_id = 0

        tokens.append(token_id)
        x = torch.cat([x, torch.tensor([[token_id]])], dim=1)

        # Stop on EOS
        if itos[token_id] == "EOS":
            break

    return tokens


# ---------- NOTES → MIDI ----------

def notes_to_midi(notes, tempo=120):
    """Convert a note list to a pretty_midi.PrettyMIDI object.

    Args:
        notes: list of (pitch, start_time, end_time, velocity)
        tempo: BPM

    Returns:
        pretty_midi.PrettyMIDI
    """
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    for (pitch, start, end, vel) in notes:
        # ensure positive duration
        if end <= start:
            end = start + 0.05
        note = pretty_midi.Note(
            velocity=int(vel),
            pitch=int(pitch),
            start=float(start),
            end=float(end),
        )
        piano.notes.append(note)

    pm.instruments.append(piano)
    return pm


# ---------- MAIN ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (baseline.pt or policy_rl.pt)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="output.wav",
        help="Output WAV file name",
    )
    ap.add_argument(
        "--length",
        type=int,
        default=256,
        help="Max number of tokens to generate",
    )
    ap.add_argument(
        "--tempo",
        type=int,
        default=120,
        help="MIDI tempo (BPM)",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    args = ap.parse_args()

    # 1. Load model (MUST MATCH TRAINING HYPERPARAMETERS)
    model = TinyTransformerLM(len(VOCAB), d=512, n=6, h=8, max_len=128)
    sd = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(sd)

    # 2. Sample tokens
    toks = sample(model, T=args.length, temperature=args.temperature)
    print("Generated", len(toks), "tokens")

    # 3. Convert tokens → notes
    notes = tokens_to_notes(toks)
    print("Converted to", len(notes), "notes")

    if len(notes) == 0:
        print("No notes generated, nothing to save.")
        return

    # 4. Create MIDI
    pm = notes_to_midi(notes, tempo=args.tempo)

    midi_path = args.out.rsplit(".", 1)[0] + ".mid"
    pm.write(midi_path)
    print("Saved MIDI to:", midi_path)

    # 5. Try to synthesize audio
    try:
        # requires pyfluidsynth + system fluidsynth installed
        audio = pm.fluidsynth(fs=22050)
        sf.write(args.out, audio, 22050)
        print("Saved audio WAV to:", args.out)
    except Exception as e:
        print("Could not synthesize audio (fluidsynth not available). Error:")
        print(e)
        print("You still have the MIDI file:", midi_path)


if __name__ == "__main__":
    main()
