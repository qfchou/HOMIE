# HOMIE
## Install

```bash
conda create -n HOMIE python=3.10 -y && conda activate HOMIE
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn==2.8.3 --no-build-isolation
pip install -r requirements.txt
```
