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

## Composed retrieval tasks

Data released at 🤗 [`qfchou/HOMIE-PCR`](https://huggingface.co/datasets/qfchou/HOMIE-PCR):
| file | task | n | built from |
|---|---|---|---|
| `multi_image_to_text.json` | (qi,qi,…) → ct | 600 | ARCH Bookset captions, regrouped |
| `image_text_to_image.json` | (qi,qt) → ci | 488 | ARCH Bookset, relational clauses parsed out by GPT-5 |

Annotations only — the images are not redistributed. Download `book_set.zip` from ARCH (https://warwick.ac.uk/fac/cross_fac/tia/data/arch),
then run just these two tasks with:

```bash
ARCH_IMAGES=/path/to/book_set/images bash run_pcr.sh
```
