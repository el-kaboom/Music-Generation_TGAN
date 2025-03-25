import torch
from models.generator import MusicGenerator
from utils.midi_processor import load_midi
from music21 import stream, note, chord

generator = MusicGenerator()
generator.load_state_dict(torch.load("music_gan_project/models/generator.pth"))
generator.eval()

#music tokens
input_tokens = torch.randint(0, 50256, (1, 100))  # Random noise
generated_tokens = generator(input_tokens).argmax(dim=-1).tolist()[0]

# to MIDI format
def tokens_to_midi(tokens, output_file="generated_music.mid"):
    midi_stream = stream.Stream()
    for t in tokens:
        n = note.Note()
        n.pitch.midi = (t % 128)  # Convert to MIDI range
        midi_stream.append(n)
    midi_stream.write("midi", fp=output_file)
    print(f"Generated music saved as {output_file}")

# Save
tokens_to_midi(generated_tokens)
