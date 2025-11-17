🎵 Reinforcement Learning for Symbolic Music Generation

This project applies deep learning + reinforcement learning to generate expressive symbolic music. Two complementary systems are developed:

System 1: LSTM-based MLE + REINFORCE trained on Nottingham folk melodies

System 2: Transformer LM + RL fine-tuning trained on Maestro piano dataset

Both systems learn musical grammar from MIDI data and use RL reward shaping to improve the musicality of generated sequences.

🚀 Project Overview

Traditional ML models trained on symbolic music can imitate datasets but often lack musicality.
This project uses RL to shape the model’s creativity using aesthetic rewards:

✔ scale conformity
✔ smooth intervals
✔ melodic diversity
✔ pitch centering
✔ tonal stability (System 2)
✔ rhythmic coherence (System 2)

The hybrid MLE + RL approach produces melodies that are more structured, expressive, and pleasant.

🛠️ Features

LSTM-based music generator (System 1)

Transformer-based piano generator (System 2)

Supervised MLE pretraining

Reinforcement Learning using REINFORCE algorithm

Music-theory reward engineering

MIDI output + WAV synthesis

Visualization tools:

pianoroll

pitch distribution

interval histogram

scale-conformity curve

Demo video generation
