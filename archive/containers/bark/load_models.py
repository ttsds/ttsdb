import sys

sys.path.append("bark-vc")

import torch
from encodec import EncodecModel
from bark_hubert_quantizer.hubert_manager import HuBERTManager
from bark_hubert_quantizer.pre_kmeans_hubert import CustomHubert
from bark_hubert_quantizer.customtokenizer import CustomTokenizer
from bark import preload_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ("quantifier_V1_hubert_base_ls960_23.pth", "tokenizer_large.pth")

print("Loading HuBERT...")
hubert_model = CustomHubert(HuBERTManager.make_sure_hubert_installed(), device=device)
print("Loading Quantizer...")
quant_model = CustomTokenizer.load_from_checkpoint(
    HuBERTManager.make_sure_tokenizer_installed(model=model[0], local_file=model[1]),
    device,
)
print("Loading Encodec...")
encodec_model = EncodecModel.encodec_model_24khz()
print("Downloaded and loaded models!")
preload_models()