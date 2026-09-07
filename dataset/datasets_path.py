import pandas as pd
from typing import Dict, List
from torch.utils.data import Dataset
import json
import os


def _prompt_suffix(content_type):
    """The instruction appended to every query and candidate.
    content_type: 'image', 'sentence', 'image and sentence', 'video', 'video and sentence'
    """
    return f"\nSummarize above {content_type} in one word: "


class PathDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(PathDataset, self).__init__()
        self.image_data_path = image_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            self.items.append({
                'img': item['img'],
                'caption': item['caption']
            })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if text is not None and image is not None:
            suffix = _prompt_suffix('image and sentence')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
                {"type": "text", "text": text},
            ]
            content.append({"type": "text", "text": suffix})
        elif image is None:
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        else:
            suffix = _prompt_suffix('image')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
            ]
            content.append({"type": "text", "text": suffix})
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'image':
            message = self.construct_messages(image=item['img'])
        else:
            message = self.construct_messages(text=item['caption'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class PathvqaDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(PathvqaDataset, self).__init__()
        self.image_data_path = image_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            self.items.append({
                'img': item['image'],
                'question': item['question'],
                'answer': item['key_finding_text']
            })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if text is not None and image is not None:
            # Query: image + question (fusion)
            suffix = _prompt_suffix('image and sentence')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
                {"type": "text", "text": text},
            ]
            content.append({"type": "text", "text": suffix})
        elif image is not None:
            # Image only
            suffix = _prompt_suffix('image')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
            ]
            content.append({"type": "text", "text": suffix})
        else:
            # Text only (the answer side)
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'query':
            # image + question fused into one query
            message = self.construct_messages(image=item['img'], text=item['question'])
        else:
            # Answer side: always text only
            message = self.construct_messages(text=item['answer'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class MultiimgDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(MultiimgDataset, self).__init__()
        self.image_data_path = image_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            self.items.append({
                'img': item['img'],
                'caption': item['caption']
            })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if image is None:
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        else:
            content = [
                {"type": "image", "image": f"{self.image_data_path}/{filename}"}
                for i, filename in enumerate(image)
            ]
            suffix = _prompt_suffix('image')
            content.append({"type": "text", "text": suffix})
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'image':
            message = self.construct_messages(image=item['img'])
        else:
            message = self.construct_messages(text=item['caption'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class PathMMUDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(PathMMUDataset, self).__init__()
        self.image_data_path = image_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # For multiple choice, we need to expand each question to have one entry per option
        if type == 'option':
            for item in data:
                for opt_idx, option in enumerate(item['options']):
                    self.items.append({
                        'No': item['No'],
                        'img': item['img'],
                        'question': item['question'],
                        'option': option,
                        'option_idx': opt_idx,
                        'answer': item['answer']
                    })
        else:  # type == 'query'
            for item in data:
                self.items.append({
                    'No': item['No'],
                    'img': item['img'],
                    'question': item['question'],
                    'options': item['options'],
                    'answer': item['answer']
                })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if text is not None and image is not None:
            suffix = _prompt_suffix('image and sentence')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
                {"type": "text", "text": text},
            ]
            content.append({"type": "text", "text": suffix})
        elif image is None:
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        else:
            suffix = _prompt_suffix('image')
            content = [
                {"type": "image", "image": self.image_data_path + '/' + image},
            ]
            content.append({"type": "text", "text": suffix})
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'query':
            message = self.construct_messages(image=item['img'], text=item['question'])
        else:  # type == 'option'
            message = self.construct_messages(text=item['option'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class ZeroshotDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        prompt_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(ZeroshotDataset, self).__init__()
        self.image_data_path = image_data_path
        self.type = type
        self.mode = mode

        if type == 'image':
            # Load test data CSV
            df = pd.read_csv(data_path)
            self.items = []
            for _, row in df.iterrows():
                self.items.append({
                    'image_name': row['image_name'],
                    'label': row['label']
                })
        else:  # type == 'prompt'
            # Load prompts JSON
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)

            # Extract templates and classnames from the first entry
            first_key = list(prompt_data.keys())[0]
            self.classnames = prompt_data[first_key]['classnames']
            self.templates = prompt_data[first_key]['templates']

            # Create all prompt combinations
            self.items = []
            self.class_to_idx = {}

            for class_idx, (class_label, class_name_list) in enumerate(self.classnames.items()):
                self.class_to_idx[class_label] = class_idx
                for classname in class_name_list:
                    for template in self.templates:
                        text = template.replace('CLASSNAME', classname)
                        self.items.append({
                            'text': text,
                            'class_label': class_label,
                            'class_idx': class_idx
                        })

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if image is not None:
            suffix = _prompt_suffix('image')
            content = [
                {"type": "image", "image": os.path.join(self.image_data_path, image)},
            ]
            content.append({"type": "text", "text": suffix})
        else:
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'image':
            message = self.construct_messages(image=item['image_name'])
        else:  # type == 'prompt'
            message = self.construct_messages(text=item['text'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class VideoPathDataset(Dataset):
    def __init__(
        self,
        video_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(VideoPathDataset, self).__init__()
        self.video_data_path = video_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            diagnosis = item['diagnosis']

            self.items.append({
                'video_name': item['video_name'],
                'diagnosis': diagnosis
            })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, video=None):
        if video is not None and text is not None:
            suffix = _prompt_suffix('video and sentence')
            content = [
                {"type": "video", "video": self.video_data_path + '/' + video + '.mp4', 'nframes': 10},
                {"type": "text", "text": text},
            ]
            content.append({"type": "text", "text": suffix})
        elif video is None:
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        else:
            suffix = _prompt_suffix('video')
            content = [
                {"type": "video", "video": self.video_data_path + '/' + video + '.mp4', 'nframes': 10},
            ]
            content.append({"type": "text", "text": suffix})
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'video':
            message = self.construct_messages(video=item['video_name'])
        else:  # type == 'text'
            message = self.construct_messages(text=item['diagnosis'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i

class ComposedRetrievalDataset(Dataset):
    def __init__(
        self,
        image_data_path: str,
        data_path: str,
        type: str,
        mode: str='pretrained',
    ) -> None:
        super(ComposedRetrievalDataset, self).__init__()
        self.image_data_path = image_data_path
        self.items = []

        # Load data from JSON file
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if type == 'query':
            # For query: source_image + modifier
            # Also build a mapping from target_image to its index in the candidate set
            target_images = sorted(set(item['target_image'] for item in data))
            self.target_to_idx = {img: idx for idx, img in enumerate(target_images)}

            for item in data:
                self.items.append({
                    'source_image': item['source_image'],
                    'modifier': item['modifier'],
                    'target_image': item['target_image'],
                    'target_idx': self.target_to_idx[item['target_image']]
                })
        else:  # type == 'target'
            # For candidates: collect all unique target images
            target_images = set()
            for item in data:
                target_images.add(item['target_image'])

            for target_image in sorted(target_images):
                self.items.append({
                    'target_image': target_image
                })

        self.type = type
        self.mode = mode

    def __len__(self) -> int:
        return len(self.items)

    def construct_messages(self, text=None, image=None):
        if text is not None and image is not None:
            # Query: source image + modifier text
            suffix = _prompt_suffix('image and sentence')
            content = [
                {"type": "image", "image": os.path.join(self.image_data_path, image)},
                {"type": "text", "text": text},
            ]
            content.append({"type": "text", "text": suffix})
        elif image is not None:
            # Image only (the target side)
            suffix = _prompt_suffix('image')
            content = [
                {"type": "image", "image": os.path.join(self.image_data_path, image)},
            ]
            content.append({"type": "text", "text": suffix})
        else:
            # Text only
            suffix = _prompt_suffix('sentence')
            content = [{"type": "text", "text": f"{text}{suffix}"}]
        message = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
        ]
        return message

    def get_instance(self, index):
        item = self.items[index]
        if self.type == 'query':
            # source image + modifier fused into one query
            message = self.construct_messages(
                image=item['source_image'],
                text=item['modifier']
            )
        else:  # type == 'target'
            message = self.construct_messages(image=item['target_image'])
        return message

    def __getitem__(self, i) -> Dict[str, List]:
        return self.get_instance(i), i
