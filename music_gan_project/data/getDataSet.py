import requests
import os
from music_gan_project.utils.midi_processor import load_midi, process_midi_dataset

#Download midi
def download_midi(url, save_path):
   
    response = requests.get(url)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(response.content)


#Process midi
def process_downloaded_midi():
   
    midi_url = "https://www.mfiles.co.uk/midi-files/mozart-symphony40-1.mid"
    save_path = "music_gan_project/data/raw/sample.mid"
    download_midi(midi_url, save_path)
    process_midi_dataset("music_gan_project/data/raw", "music_gan_project/data/processed/notes.pkl")

if __name__ == "__main__":
    process_downloaded_midi()
