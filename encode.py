import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

from collators.eval_collator import EvalDataCollator
from dataset.datasets_path import _prompt_suffix
from models.qwen3_vl import Qwen3VLRetForConditionalGeneration

EMB_TOKEN = "<emb>"


def build_message(image=None, text=None, video=None, nframes=10):
    """Build one chat message, identical to the datasets in dataset/datasets_path.py.

    image: a path, or a list of paths (multi-image query)
    video: a path to an .mp4
    text : a caption / question / modifier
    Any combination is allowed; image+text is the composed (interleaved) query.
    """
    if video is not None:
        content = [{"type": "video", "video": video, "nframes": nframes}]
        content_type = "video and sentence" if text else "video"
        if text:
            content.append({"type": "text", "text": text})
    elif image is not None:
        images = [image] if isinstance(image, str) else list(image)
        content = [{"type": "image", "image": im} for im in images]
        content_type = "image and sentence" if text else "image"
        if text:
            content.append({"type": "text", "text": text})
    elif text is not None:
        content, content_type = None, "sentence"
    else:
        raise ValueError("build_message needs at least one of image / text / video")

    suffix = _prompt_suffix(content_type)
    if content is None:  # text-only: the suffix is glued onto the text itself
        content = [{"type": "text", "text": f"{text}{suffix}"}]
    else:
        content.append({"type": "text", "text": suffix})

    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": f"{EMB_TOKEN}."}]},
    ]


class _MessageDataset(Dataset):
    """Feeds (message, id) pairs to EvalDataCollator, like the eval datasets do."""

    def __init__(self, messages):
        self.messages = messages

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, i):
        return self.messages[i], i


class Encoder:
    def __init__(
        self,
        model_id,
        original_model_id=None,
        device=None,
        torch_dtype=torch.bfloat16,
    ):
        """model_id: the merged retrieval checkpoint -- a Hub repo id
        (e.g. qfchou/HOMIE-Qwen3-VL-8B) or a local directory holding the same files.
        original_model_id: where the processor comes from; defaults to model_id.
        The training run never changes the processor, so the base repo
        (Qwen/Qwen3-VL-8B-Instruct) works too.
        """
        self.model = Qwen3VLRetForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(original_model_id or model_id)
        tokenizer = self.processor.tokenizer

        # same as add_embed_token() in eval_path.py. The released checkpoints were
        # saved after the token was appended, so a tokenizer that already carries
        # <emb> is fine -- both paths end at vocab_size = 151670.
        tokenizer.add_tokens([EMB_TOKEN])
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.config.emb_token_ids = tokenizer.convert_tokens_to_ids([EMB_TOKEN])

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.collator = EvalDataCollator(tokenizer=tokenizer, processor=self.processor)

    def _to_device(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if key in ("pixel_values", "pixel_values_videos"):
                    batch[key] = value.to(self.device).to(self.model.dtype)
                else:
                    batch[key] = value.to(self.device)
        return batch

    @torch.no_grad()
    def encode(self, items, batch_size=8, num_workers=4, normalize=True):
        """items: list of dicts accepted by build_message, e.g.
        {"text": "..."} / {"image": "a.jpg"} / {"image": "a.jpg", "text": "..."}
        / {"image": ["a.jpg", "b.jpg"]} / {"video": "case.mp4"}
        Returns a float32 tensor of shape (len(items), hidden_size).
        """
        messages = [build_message(**item) for item in items]
        loader = DataLoader(
            _MessageDataset(messages),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=self.collator,
        )

        features = []
        for batch in loader:
            batch = self._to_device(batch)
            embeddings, _ = self.model(**batch, inference=True)
            features.append(embeddings.float().cpu())

        features = torch.cat(features)
        return F.normalize(features, dim=-1) if normalize else features


def _load_candidates(path):
    """A .json list of strings, or of dicts like {"image": ..., "text": ...}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [{"text": item} if isinstance(item, str) else item for item in data]


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against one composed query.")
    parser.add_argument("--model_id", required=True, help="merged retrieval checkpoint")
    parser.add_argument("--original_model_id", default=None, help="processor source (default: --model_id)")
    parser.add_argument("--image", default=None, help="query image (repeatable via comma)")
    parser.add_argument("--text", default=None, help="query text / question / modifier")
    parser.add_argument("--video", default=None, help="query video (.mp4)")
    parser.add_argument("--candidates", required=True, help="json file with the candidate pool")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    query = {}
    if args.image:
        images = [p for p in args.image.split(",") if p]
        query["image"] = images[0] if len(images) == 1 else images
    if args.text:
        query["text"] = args.text
    if args.video:
        query["video"] = args.video
    if not query:
        parser.error("give at least one of --image / --text / --video")

    encoder = Encoder(args.model_id, args.original_model_id)
    candidates = _load_candidates(args.candidates)

    query_embedding = encoder.encode([query], batch_size=1)
    candidate_embeddings = encoder.encode(candidates, batch_size=args.batch_size)
    scores = (query_embedding @ candidate_embeddings.T)[0]

    topk = torch.topk(scores, k=min(args.topk, len(candidates)))
    print(f"\nquery: {query}\n")
    for rank, (score, index) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), start=1):
        print(f"{rank:2d}. {score:.4f}  {candidates[index]}")


if __name__ == "__main__":
    main()
