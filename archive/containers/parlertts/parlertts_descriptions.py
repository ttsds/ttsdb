from pathlib import Path

import wespeaker
from datasets import load_dataset
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import torch

libritts_r = load_dataset("mythicinfinity/libritts_r", "other", split="test.other", streaming=True)

Path("/tmp/libritts_r").mkdir(exist_ok=True)
for ex in tqdm(libritts_r, desc="Downloading audio"):
    name = Path(ex["path"]).stem
    sf.write(f"/tmp/libritts_r/{name}.wav", ex["audio"]["array"], ex["audio"]["sampling_rate"])

ds = load_dataset("parler-tts/libritts-r-filtered-speaker-descriptions", "other", split="test.other", streaming=True)
emb_list = []

wespeaker_model = wespeaker.load_model('english')

Path("/tmp/embeddings").mkdir(exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
wespeaker_model.set_device(device)
# shuffle the dataset
ds = ds.shuffle(1000)
i = 0
max_i = 100
for ex in tqdm(ds, desc="Getting embeddings"):
    if i >= max_i:
        break
    i += 1
    name = Path(ex["path"]).stem
    emb = wespeaker_model.extract_embedding(f"/tmp/libritts_r/{name}.wav")
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    np.save(f"/tmp/embeddings/{name}.npy", emb)
    with open(f"/tmp/embeddings/{name}.txt", "w") as f:
        f.write(ex["text_description"])