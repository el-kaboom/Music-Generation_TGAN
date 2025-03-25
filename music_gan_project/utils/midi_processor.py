# MIDI Processing 
import os
import pickle
from music21 import converter, note, chord

def load_midi(file_path):
    midi = converter.parse(file_path)
    notes = []

    for part in midi.parts:
        for element in part.recurse():
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))  
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))  
    
    return notes

def process_midi_dataset(input_folder, output_file):
    all_notes = []
    midi_files = [file for file in os.listdir(input_folder) if file.endswith(".mid")]

    for file in midi_files:
        file_path = os.path.join(input_folder, file)
        print(f"Processing {file_path}...")
        notes = load_midi(file_path)
        all_notes.extend(notes)


    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(all_notes, f)



