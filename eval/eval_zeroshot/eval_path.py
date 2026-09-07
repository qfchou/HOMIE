import os
from transformers import AutoProcessor
import sys
current_file_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(current_file_path, "../../")
sys.path.append(module_path)
from models.qwen3_vl import Qwen3VLRetForConditionalGeneration
import torch
import argparse
from dataset.datasets_path import PathDataset, PathvqaDataset, MultiimgDataset, PathMMUDataset, ZeroshotDataset, VideoPathDataset, ComposedRetrievalDataset
from collators.eval_collator import EvalDataCollator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
import accelerate
import json
import numpy as np
import csv
from datetime import datetime

def save_results_to_csv(csv_path, model_id, task, data_path, metrics, args):
    """Save evaluation results to CSV file. Create if not exists, append if exists."""
    file_exists = os.path.isfile(csv_path)

    # Prepare row data
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_id': model_id,
        'original_model_id': args.original_model_id,
        'task': task,
        'data_path': data_path,
        'image_data_path': args.image_data_path if hasattr(args, 'image_data_path') else None,
        'video_data_path': args.video_data_path if hasattr(args, 'video_data_path') else None,
        'batch_size': args.batch_size,
        'mode': args.mode,
    }

    # Add all metrics
    row_data.update(metrics)

    # If file exists, read existing fieldnames and merge with new ones
    if file_exists:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fieldnames = reader.fieldnames if reader.fieldnames else []

        # Merge fieldnames: keep existing order and append new columns at the end
        all_fieldnames = list(existing_fieldnames)
        for key in row_data.keys():
            if key not in all_fieldnames:
                all_fieldnames.append(key)

        # Check if new columns were added
        if len(all_fieldnames) > len(existing_fieldnames):
            # Read all existing data first
            with open(csv_path, 'r', newline='', encoding='utf-8') as read_f:
                reader = csv.DictReader(read_f)
                existing_rows = list(reader)

            # Rewrite file with new header and all data
            with open(csv_path, 'w', newline='', encoding='utf-8') as write_f:
                writer = csv.DictWriter(write_f, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row_data)
        else:
            # No new columns, just append
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writerow(row_data)
    else:
        # File doesn't exist, create new file with header and data
        all_fieldnames = list(row_data.keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerow(row_data)

    print(f"\nResults saved to {csv_path}")

def eval(args):
    original_model_id = args.original_model_id
    model_id = args.model_id
    model = Qwen3VLRetForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # processor is not changed so we still load from the original model repo
    processor = AutoProcessor.from_pretrained(original_model_id)

    tokenizer = processor.tokenizer

    def add_embed_token(tokenizer, model, emb_token="<emb>"):
        # A tokenizer that already carries <emb> is fine: add_tokens() is then a
        # no-op and resize_token_embeddings() keeps the checkpoint's 151670 rows.
        tokenizer.add_tokens([emb_token])

        model.resize_token_embeddings(len(tokenizer))

        emb_token_ids = tokenizer.convert_tokens_to_ids([emb_token])
        model.config.emb_token_ids = emb_token_ids

    add_embed_token(tokenizer, model)

    if args.task=='retrieval':
        query_dataset = PathDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='image',
            mode=args.mode
        )

        cand_dataset = PathDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='text',
            mode=args.mode
        )
    elif args.task=='vqa':
        query_dataset = PathvqaDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='query',
            mode=args.mode
        )

        cand_dataset = PathvqaDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='answer',
            mode=args.mode
        )
    elif args.task=='multire':
        query_dataset = MultiimgDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='image',
            mode=args.mode
        )

        cand_dataset = MultiimgDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='text',
            mode=args.mode
        )
    elif args.task=='mcq':
        query_dataset = PathMMUDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='query',
            mode=args.mode
        )

        cand_dataset = PathMMUDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='option',
            mode=args.mode
        )
    elif args.task=='zeroshot':
        query_dataset = ZeroshotDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            prompt_path=args.prompt_path,
            type='image',
            mode=args.mode
        )

        cand_dataset = ZeroshotDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            prompt_path=args.prompt_path,
            type='prompt',
            mode=args.mode
        )
    elif args.task=='videoretrieval':
        query_dataset = VideoPathDataset(
            video_data_path=args.video_data_path,
            data_path=args.data_path,
            type='video',
            mode=args.mode
        )

        cand_dataset = VideoPathDataset(
            video_data_path=args.video_data_path,
            data_path=args.data_path,
            type='text',
            mode=args.mode
        )
    elif args.task=='composedretrieval':
        query_dataset = ComposedRetrievalDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='query',
            mode=args.mode
        )

        cand_dataset = ComposedRetrievalDataset(
            image_data_path=args.image_data_path,
            data_path=args.data_path,
            type='target',
            mode=args.mode
        )

    query_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_data_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)

    query_dataloader = DataLoader(query_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=query_data_collator)
    candidate_dataloader = DataLoader(cand_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn=cand_data_collator)

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device
    is_main_process = accelerator.is_main_process

    model.eval()

    def tensors_to_device(data, device, dtype=model.dtype):
        for key in data.keys():
            if isinstance(data[key], torch.Tensor):
                if key == 'pixel_values' or key == 'pixel_values_videos':
                    data[key] = data[key].to(device).to(dtype)
                else:
                    data[key] = data[key].to(device)
        return data

    query_features = []
    query_ids = []
    candidate_features = []
    candidate_ids = []

    from tqdm import tqdm
    with torch.no_grad():
        query_dataloader, candidate_dataloader, model = accelerator.prepare(query_dataloader, candidate_dataloader, model)

        for batch in tqdm(query_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            query_embed, batch_query_ids = model(**batch, inference=True)
            query_embed = F.normalize(query_embed, dim=-1)
            query_embed = accelerator.gather_for_metrics(query_embed)
            batch_query_ids = accelerate.utils.gather_object(batch_query_ids)[:len(query_embed)]
            query_ids.extend(batch_query_ids)
            query_features.append(query_embed)

        for batch in tqdm(candidate_dataloader, disable=not is_main_process):
            batch = tensors_to_device(batch, device)
            candidate_embed, batch_candidate_ids = model(**batch, inference=True)
            candidate_embed = F.normalize(candidate_embed, dim=-1)
            candidate_embed = accelerator.gather_for_metrics(candidate_embed)
            batch_candidate_ids = accelerator.gather_for_metrics(batch_candidate_ids)[:len(candidate_embed)]
            candidate_ids.extend(batch_candidate_ids)
            candidate_features.append(candidate_embed)

    query_features = torch.cat(query_features, dim=0)
    candidate_features = torch.cat(candidate_features, dim=0)

    if is_main_process:
        # Adjust the order according to ids
        query_ids = np.array(query_ids)
        sorted_query_indices = np.argsort(query_ids)
        image_features = query_features[sorted_query_indices]
        candidate_ids = np.array(candidate_ids)
        sorted_candidate_indices = np.argsort(candidate_ids)
        text_features = candidate_features[sorted_candidate_indices]

        # Save embeddings if requested
        if args.embeddings_output:
            embeddings_dict = {
                'query_embeddings': image_features.cpu().numpy(),
                'query_ids': query_ids,
                'candidate_embeddings': text_features.cpu().numpy(),
                'candidate_ids': candidate_ids,
                'task': args.task,
                'model_id': args.model_id,
                'data_path': args.data_path
            }
            np.savez(args.embeddings_output, **embeddings_dict)
            print(f"\nEmbeddings saved to {args.embeddings_output}")
            print(f"  Query embeddings shape: {image_features.shape}")
            print(f"  Candidate embeddings shape: {text_features.shape}")

        if args.task == 'mcq':
            # For multiple choice, we need to compute accuracy
            with open(args.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            correct = 0
            total = len(data)

            for idx, item in enumerate(data):
                query_emb = image_features[idx:idx+1]

                num_options = len(item['options'])
                option_start_idx = idx * num_options
                option_end_idx = option_start_idx + num_options
                option_embs = text_features[option_start_idx:option_end_idx]

                scores = (query_emb @ option_embs.t()).squeeze()

                predicted_idx = scores.argmax().item()
                predicted_answer = item['options'][predicted_idx]

                if predicted_answer == item['answer']:
                    correct += 1

            accuracy = correct / total
            metrics = {'accuracy': accuracy}
            print(f"Multiple Choice Accuracy: {accuracy:.4f} ({correct}/{total})")
            print(metrics)

            save_results_to_csv(args.csv_output, args.model_id, args.task, args.data_path, metrics, args)
        elif args.task == 'zeroshot':
            # Zero-shot classification with prompt ensembling
            import pandas as pd
            from sklearn.metrics import balanced_accuracy_score

            df = pd.read_csv(args.data_path)
            true_labels = df['label'].values

            classnames = cand_dataset.classnames
            templates = cand_dataset.templates
            class_to_idx = cand_dataset.class_to_idx

            num_classes = len(classnames)
            num_classnames_per_class = [len(names) for names in classnames.values()]
            num_templates = len(templates)

            # Build class embeddings with prompt ensembling
            class_embeddings = []
            text_idx = 0

            for class_label, class_name_list in classnames.items():
                embeddings_for_class = []

                for classname in class_name_list:
                    classname_embeddings = text_features[text_idx:text_idx + num_templates]
                    embeddings_for_class.append(classname_embeddings)
                    text_idx += num_templates

                class_embedding = torch.stack(embeddings_for_class, dim=0)
                class_embedding = class_embedding.mean(dim=(0, 1))
                class_embedding = class_embedding / class_embedding.norm()

                class_embeddings.append(class_embedding)

            class_embeddings = torch.stack(class_embeddings, dim=0)

            scores = image_features @ class_embeddings.t()

            predictions = scores.argmax(dim=1).cpu().numpy()

            idx_to_class = {idx: label for label, idx in class_to_idx.items()}
            predicted_labels = [idx_to_class[pred] for pred in predictions]

            balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)
            accuracy = (predictions == np.array([class_to_idx[label] for label in true_labels])).mean()

            metrics = {
                'balanced_accuracy': balanced_acc,
                'accuracy': accuracy
            }

            print(f"Zero-shot Classification Results:")
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            print(metrics)

            save_results_to_csv(args.csv_output, args.model_id, args.task, args.data_path, metrics, args)
        elif args.task == 'composedretrieval':
            # Composed retrieval evaluation: source_image + modifier -> target_image
            query_to_target = []
            for item in query_dataset.items:
                query_to_target.append(item['target_idx'])
            query_to_target = torch.tensor(query_to_target, device=device)

            assert text_features.isnan().sum().item() == 0, 'nan in target image emb'
            assert image_features.isnan().sum().item() == 0, 'nan in query emb'

            scores = image_features @ text_features.t()

            positive_pairs = torch.zeros_like(scores, dtype=bool)
            positive_pairs[torch.arange(len(scores)), query_to_target] = True

            metrics = {}
            recall_k_list = [1, 5, 10]
            batch_size = 64

            for recall_k in recall_k_list:
                metrics[f"query_retrieval_recall@{recall_k}"] = (
                    batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k) > 0
                ).float().mean().item()

            print(f"Composed Retrieval Results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            if args.save_detailed_results:
                detailed_results = []
                top_k = 10

                for idx in range(len(scores)):
                    topk_scores, topk_indices = torch.topk(scores[idx], k=top_k, dim=0)

                    ground_truth_idx = query_to_target[idx].item()

                    item = query_dataset.items[idx]
                    modifier = item.get('modifier', item.get('caption', ''))

                    topk_indices_list = topk_indices.cpu().tolist()
                    is_correct_at_k = {
                        'recall@1': 1 if ground_truth_idx in topk_indices_list[:1] else 0,
                        'recall@5': 1 if ground_truth_idx in topk_indices_list[:5] else 0,
                        'recall@10': 1 if ground_truth_idx in topk_indices_list[:10] else 0
                    }

                    result_entry = {
                        'query_index': idx,
                        'modifier': modifier,
                        'ground_truth_index': ground_truth_idx,
                        'predicted_indices': topk_indices_list,
                        'predicted_scores': topk_scores.cpu().tolist(),
                        'is_correct_at_k': is_correct_at_k
                    }
                    detailed_results.append(result_entry)

                output_base = args.csv_output.replace('.csv', '')
                detailed_output_file = f"{output_base}_detailed.json"

                with open(detailed_output_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, indent=2, ensure_ascii=False)

                print(f"\nDetailed results saved to: {detailed_output_file}")
                print(f"Total queries: {len(detailed_results)}")

            save_results_to_csv(args.csv_output, args.model_id, args.task, args.data_path, metrics, args)
        else:
            # Original retrieval evaluation
            texts_image_index = list(range(image_features.shape[0]))

            assert text_features.isnan().sum().item() == 0, 'nan in retrieve emb'
            assert image_features.isnan().sum().item() == 0, 'nan in images emb'

            scores  = text_features @ image_features.t()

            positive_pairs = torch.zeros_like(scores, dtype=bool)
            positive_pairs[torch.arange(len(scores)), texts_image_index] = True
            metrics = {}
            recall_k_list = [1, 5, 10]
            batch_size = 64
            if args.task == 'retrieval':
                for recall_k in recall_k_list:
                    metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()
                    metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item()

                if args.save_detailed_results:
                    top_k = 10

                    with open(args.data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Text-to-Image retrieval results
                    t2i_results = []
                    for idx in range(len(scores)):
                        topk_scores, topk_indices = torch.topk(scores[idx], k=top_k, dim=0)

                        ground_truth_idx = texts_image_index[idx]

                        item = data[idx]
                        query_text = item.get('text', item.get('caption', ''))

                        topk_indices_list = topk_indices.cpu().tolist()
                        is_correct_at_k = {
                            'recall@1': 1 if ground_truth_idx in topk_indices_list[:1] else 0,
                            'recall@5': 1 if ground_truth_idx in topk_indices_list[:5] else 0,
                            'recall@10': 1 if ground_truth_idx in topk_indices_list[:10] else 0
                        }

                        result_entry = {
                            'query_index': idx,
                            'query_text': query_text,
                            'ground_truth_index': ground_truth_idx,
                            'predicted_indices': topk_indices_list,
                            'predicted_scores': topk_scores.cpu().tolist(),
                            'is_correct_at_k': is_correct_at_k
                        }
                        t2i_results.append(result_entry)

                    # Image-to-Text retrieval results
                    i2t_results = []
                    scores_i2t = scores.T
                    for idx in range(len(scores_i2t)):
                        topk_scores, topk_indices = torch.topk(scores_i2t[idx], k=top_k, dim=0)

                        ground_truth_idx = idx

                        item = data[idx]
                        query_image = item.get('image', item.get('image_id', f'image_{idx}'))

                        topk_indices_list = topk_indices.cpu().tolist()
                        is_correct_at_k = {
                            'recall@1': 1 if ground_truth_idx in topk_indices_list[:1] else 0,
                            'recall@5': 1 if ground_truth_idx in topk_indices_list[:5] else 0,
                            'recall@10': 1 if ground_truth_idx in topk_indices_list[:10] else 0
                        }

                        result_entry = {
                            'query_index': idx,
                            'query_image': query_image,
                            'ground_truth_index': ground_truth_idx,
                            'predicted_indices': topk_indices_list,
                            'predicted_scores': topk_scores.cpu().tolist(),
                            'is_correct_at_k': is_correct_at_k
                        }
                        i2t_results.append(result_entry)

                    output_base = args.csv_output.replace('.csv', '')
                    t2i_output_file = f"{output_base}_t2i_detailed.json"
                    i2t_output_file = f"{output_base}_i2t_detailed.json"

                    with open(t2i_output_file, 'w', encoding='utf-8') as f:
                        json.dump(t2i_results, f, indent=2, ensure_ascii=False)

                    with open(i2t_output_file, 'w', encoding='utf-8') as f:
                        json.dump(i2t_results, f, indent=2, ensure_ascii=False)

                    print(f"\nDetailed results saved:")
                    print(f"  Text-to-Image: {t2i_output_file} ({len(t2i_results)} queries)")
                    print(f"  Image-to-Text: {i2t_output_file} ({len(i2t_results)} queries)")

            else:
                for recall_k in recall_k_list:
                    metrics[f"query_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item()

            print(metrics)

            if args.save_detailed_results and args.task in ['vqa', 'multire']:
                detailed_results = []
                top_k = 10

                with open(args.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for idx in range(len(scores)):
                    topk_scores, topk_indices = torch.topk(scores[idx], k=top_k, dim=0)

                    ground_truth_idx = texts_image_index[idx]

                    item = data[idx]
                    if args.task == 'vqa':
                        query_text = item.get('question', '')
                        ground_truth_text = item.get('answer', '')
                    else:  # multire
                        query_text = item.get('caption', item.get('text', ''))
                        ground_truth_text = query_text

                    topk_indices_list = topk_indices.cpu().tolist()
                    is_correct_at_k = {
                        'recall@1': 1 if ground_truth_idx in topk_indices_list[:1] else 0,
                        'recall@5': 1 if ground_truth_idx in topk_indices_list[:5] else 0,
                        'recall@10': 1 if ground_truth_idx in topk_indices_list[:10] else 0
                    }

                    result_entry = {
                        'query_index': idx,
                        'query_text': query_text,
                        'ground_truth_index': ground_truth_idx,
                        'ground_truth_text': ground_truth_text,
                        'predicted_indices': topk_indices_list,
                        'predicted_scores': topk_scores.cpu().tolist(),
                        'is_correct_at_k': is_correct_at_k
                    }
                    detailed_results.append(result_entry)

                output_base = args.csv_output.replace('.csv', '')
                detailed_output_file = f"{output_base}_detailed.json"

                with open(detailed_output_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, indent=2, ensure_ascii=False)

                print(f"\nDetailed results saved to: {detailed_output_file}")
                print(f"Total queries: {len(detailed_results)}")

            save_results_to_csv(args.csv_output, args.model_id, args.task, args.data_path, metrics, args)

def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k

def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_data_path', type=str)
    parser.add_argument('--video_data_path', type=str, default=None, help='Path to video data directory')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--original_model_id', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--batch_size', type=int, default=7)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_path', type=str, default=None)
    parser.add_argument('--mode', type=str, default='pretrained')
    parser.add_argument('--csv_output', type=str, default='eval_results.csv', help='Path to CSV file for saving results')
    parser.add_argument('--embeddings_output', type=str, default=None,
                        help='Path to save embeddings (default: embeddings.npz)')
    parser.add_argument('--save_detailed_results', action='store_true',
                        help='Save detailed results for each query including top-k predictions and scores')

    args = parser.parse_args()
    eval(args)
