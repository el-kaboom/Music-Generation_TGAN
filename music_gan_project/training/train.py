import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from models.generator import MusicGenerator
from models.discriminator import MusicDiscriminator

# Load Processed MIDI Data
with open("music_gan_project/data/processed/notes.pkl", "rb") as f:
    notes = pickle.load(f)

print("Loaded", len(notes), "notes for training!")

# Convert Notes to Integers
unique_notes = sorted(set(notes))
note_to_int = {note: num for num, note in enumerate(unique_notes)}
int_to_note = {num: note for note, num in note_to_int.items()}

# Create Training Sequences
sequence_length = 100 
input_sequences = []
output_notes = []

for i in range(len(notes) - sequence_length):
    input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
    output_notes.append(note_to_int[notes[i + sequence_length]])

# Convert to PyTorch tensors
input_sequences = torch.tensor(input_sequences, dtype=torch.long)
output_notes = torch.tensor(output_notes, dtype=torch.long)

# Define Models
generator = MusicGenerator(input_dim=len(unique_notes))
discriminator = MusicDiscriminator(input_dim=len(unique_notes))

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
disc_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)

# Loss Function
criterion = nn.BCELoss()

# Training Loop
num_epochs = 50
batch_size = 64

for epoch in range(num_epochs):
    gen_losses = []
    disc_losses = []

    for i in range(0, len(input_sequences), batch_size):
        # Prepare Data Batch
        batch_inputs = input_sequences[i:i + batch_size]
        batch_outputs = output_notes[i:i + batch_size]

        # Convert target outputs to one-hot vectors
        target_one_hot = torch.zeros((batch_size, len(unique_notes)))
        for j in range(batch_size):
            target_one_hot[j, batch_outputs[j]] = 1

       
        # Train Discriminator
        real_labels = torch.ones(batch_size, 1)  # Real = 1
        fake_labels = torch.zeros(batch_size, 1)  # Fake = 0

        # Get real predictions
        real_preds = discriminator(target_one_hot)
        real_loss = criterion(real_preds, real_labels)

        # Generate Fake Data
        noise = torch.randint(0, len(unique_notes), (batch_size, sequence_length))
        fake_music = generator(noise)
        fake_preds = discriminator(fake_music.detach())  # Stop backprop to generator
        fake_loss = criterion(fake_preds, fake_labels)

        # Discriminator Loss
        disc_loss = (real_loss + fake_loss) / 2

        disc_optimizer.zero_grad()
        disc_loss.backward()
        disc_optimizer.step()

        
        # Train Generator
        fake_preds = discriminator(fake_music)
        gen_loss = criterion(fake_preds, real_labels)  # Try to fool discriminator

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        gen_losses.append(gen_loss.item())
        disc_losses.append(disc_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs} - Generator Loss: {np.mean(gen_losses):.4f}, Discriminator Loss: {np.mean(disc_losses):.4f}")


os.makedirs("music_gan_project/models", exist_ok=True)
torch.save(generator.state_dict(), "music_gan_project/models/generator.pth")
torch.save(discriminator.state_dict(), "music_gan_project/models/discriminator.pth")

print("Hogaya Train")
