# HOMIE - Histopathology Multi-modal Embedding for Pathology Composed Retrieval (ECCV26)
🤗 [Models & data](https://huggingface.co/collections/qfchou/homie)
## Install

```bash
conda create -n HOMIE python=3.10 -y && conda activate HOMIE
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
```

## Quick start

```python
from encode import Encoder

enc = Encoder("qfchou/HOMIE-Qwen3-VL-2B")
emb = enc.encode([{"image": "slide.jpg"}])     # torch.Tensor, (1, 2048), L2-normalised
```

`encode()` takes a list of dicts and returns one row per item:

```python
{"text": "invasive ductal carcinoma"}                 # text only
{"image": "slide.jpg"}                                # single image
{"image": ["a.png", "b.png"]}                         # multi-image query
{"image": "slide.jpg", "text": "is there necrosis?"}  # composed (interleaved) query
{"video": "case.mp4"}                                 # video
{"video": "case.mp4", "text": "best diagnosis?"}      # video + text
```
