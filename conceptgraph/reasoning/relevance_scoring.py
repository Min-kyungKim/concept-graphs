"""
Task-Conditioned Relevance Scoring Module

Input:
    - O_T: ConceptGraphs 결과물 (pkl.gz) — 각 객체 o_j는 f_j (CLIP feature), p_j (pointcloud) 보유
    - scenario.json — tasks 리스트

Output:
    - s_j: 각 객체의 relevance score in [0, 1]

Method:
    - 각 task 텍스트를 CLIP text encoder로 임베딩
    - f_j (CLIP image embedding)와 각 task embedding 간 cosine similarity 계산
    - 객체별로 모든 task similarity 중 max를 s_j로 사용
"""

import csv
import gzip
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import open_clip
import torch
import torch.nn.functional as F


def load_clip_model(model_name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k") -> Tuple[torch.nn.Module, object, str]:
    """Load the CLIP model used by ConceptGraphs."""
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), tokenizer, device


def extract_tasks(scenario_json_path: str, scenario_idx: int = 0, route_key: Optional[str] = None) -> List[str]:
    """Extract the task texts from scenario.json."""
    with open(scenario_json_path, "r") as f:
        data = json.load(f)

    scenario_key = str(scenario_idx + 1)
    scenario = data["scenario"][scenario_key]

    tasks: List[str] = []
    for route_id, route in scenario["route"].items():
        if route_key is not None and route_id != route_key:
            continue
        tasks.extend(route["task"])

    print(f"[Scenario {scenario_key}] goal: {scenario['scenario_goal']}")
    print(f"[Tasks] {tasks}")
    return tasks


@torch.no_grad()
def embed_tasks(tasks: List[str], model, tokenizer, device: str) -> torch.Tensor:
    """Encode task text into CLIP text embeddings."""
    tokenized = tokenizer(tasks).to(device)
    text_embeddings = model.encode_text(tokenized)
    return F.normalize(text_embeddings, dim=-1)


def load_objects_from_conceptgraph(pkl_path: str) -> List[Dict]:
    """Load objects from a ConceptGraphs pkl.gz file."""
    with gzip.open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, list):
        objects = loaded
    elif isinstance(loaded, dict) and "objects" in loaded:
        objects = loaded["objects"]
    elif isinstance(loaded, dict):
        objects = list(loaded.values())
    else:
        objects = loaded

    print(f"[ConceptGraphs] {len(objects)} objects loaded")
    return objects


def extract_normalized_object_features(objects: List[Dict], device: str) -> torch.Tensor:
    """Extract and normalize CLIP features for each object."""
    features = []
    for obj in objects:
        if "clip_ft" in obj:
            raw_feature = obj["clip_ft"]
        elif "ft" in obj:
            raw_feature = obj["ft"]
        else:
            raise KeyError(f"No feature key found in object. Available keys: {list(obj.keys())}")

        feature_tensor = (
            torch.tensor(raw_feature, dtype=torch.float32)
            if not isinstance(raw_feature, torch.Tensor)
            else raw_feature
        )

        if feature_tensor.ndim == 2:
            feature_tensor = feature_tensor.mean(dim=0)

        features.append(feature_tensor)

    object_features = torch.stack(features).to(device)
    return F.normalize(object_features, dim=-1)


@torch.no_grad()
def compute_relevance_scores(
    obj_features: torch.Tensor,
    task_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute per-object relevance scores against task embeddings."""
    similarity_matrix = obj_features @ task_embeddings.T
    best_scores, best_task_indices = similarity_matrix.max(dim=1)
    relevance_scores = (best_scores + 1) / 2
    return relevance_scores, similarity_matrix, best_task_indices


def get_object_point_count(obj: Dict) -> Optional[int]:
    pcd = obj.get("pcd_np")
    if pcd is None:
        return None
    return int(pcd.shape[0])


def format_object_summary(obj: Dict, score: float, best_task: str) -> str:
    class_name = obj.get("class_name", "N/A")
    num_detections = obj.get("num_detections", "N/A")
    n_points = get_object_point_count(obj) or "N/A"
    return (
        f"score={score:.4f} "
        f"| class_name={class_name} "
        f"| num_det={num_detections} "
        f"| n_points={n_points} "
        f"| best_task='{best_task}'"
    )


def save_relevance_scores_csv(
    objects: List[Dict],
    scores: torch.Tensor,
    best_task_indices: torch.Tensor,
    tasks: List[str],
    output_path: str,
) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    scores_cpu = scores.detach().cpu()
    best_task_indices_cpu = best_task_indices.detach().cpu()
    sorted_indices = scores_cpu.argsort(descending=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "object_idx",
            "relevance_score",
            "best_task_idx",
            "best_task",
            "num_detections",
            "n_points",
            "class_name",
        ])

        for rank, idx_tensor in enumerate(sorted_indices, start=1):
            idx = int(idx_tensor)
            obj = objects[idx]
            task_index = int(best_task_indices_cpu[idx])
            writer.writerow([
                rank,
                idx,
                float(scores_cpu[idx]),
                task_index,
                tasks[task_index],
                obj.get("num_detections", None),
                get_object_point_count(obj),
                obj.get("class_name", None),
            ])

    print(f"[Saved] relevance scores: {output_path}")


def print_top_relevance_summary(
    objects: List[Dict],
    scores: torch.Tensor,
    best_task_indices: torch.Tensor,
    tasks: List[str],
    top_k: int = 10,
) -> None:
    print("\n[Results] Top relevance scores:")
    sorted_indices = scores.argsort(descending=True)

    for rank, idx_tensor in enumerate(sorted_indices[:top_k], start=1):
        idx = int(idx_tensor)
        best_task = tasks[int(best_task_indices[idx])]
        summary = format_object_summary(objects[idx], float(scores[idx]), best_task)
        print(f"  #{rank:02d} | obj_{idx:03d} | {summary}")


def run_relevance_scoring(
    conceptgraph_path: str,
    scenario_path: str,
    scenario_idx: int = 0,
    route_key: Optional[str] = None,
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2b_s32b_b79k",
    output_csv: Optional[str] = None,
) -> Dict[str, object]:
    """Run the full task-conditioned relevance scoring pipeline."""
    print("[1] Loading CLIP model...")
    model, tokenizer, device = load_clip_model(model_name, pretrained)

    print("[2] Extracting tasks...")
    tasks = extract_tasks(scenario_path, scenario_idx, route_key)

    print("[3] Embedding tasks...")
    task_embeddings = embed_tasks(tasks, model, tokenizer, device)
    print(f"    task_embeddings shape: {task_embeddings.shape}")

    print("[4] Loading ConceptGraphs objects...")
    objects = load_objects_from_conceptgraph(conceptgraph_path)

    print("[5] Extracting object features...")
    obj_features = extract_normalized_object_features(objects, device)
    print(f"    obj_features shape: {obj_features.shape}")

    print("[6] Computing relevance scores...")
    scores, similarity_matrix, best_task_indices = compute_relevance_scores(obj_features, task_embeddings)

    print_top_relevance_summary(objects, scores, best_task_indices, tasks)

    if output_csv is not None:
        print("\n[7] Saving relevance scores to CSV...")
        save_relevance_scores_csv(
            objects=objects,
            scores=scores,
            best_task_indices=best_task_indices,
            tasks=tasks,
            output_path=output_csv,
        )

    return {
        "objects": objects,
        "scores": scores,
        "sim_matrix": similarity_matrix,
        "best_task_idx": best_task_indices,
        "tasks": tasks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_path", type=str, required=True, help="ConceptGraphs result pkl.gz path")
    parser.add_argument("--scenario_json", type=str, required=True, help="scenario.json path")
    parser.add_argument("--scenario_idx", type=int, default=0, help="Scenario index (0-based)")
    parser.add_argument("--route_key", type=str, default=None, help="Route key to use (e.g. '1.A')")
    parser.add_argument("--output_csv", type=str, default=None, help="CSV output path for relevance scores")
    args = parser.parse_args()

    run_relevance_scoring(
        conceptgraph_path=args.pkl_path,
        scenario_path=args.scenario_json,
        scenario_idx=args.scenario_idx,
        route_key=args.route_key,
        output_csv=args.output_csv,
    )
