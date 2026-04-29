import torch
import clip
from PIL import Image

class RelevanceScorer:
    def __init__(self, device="cuda", fusion="mean"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.fusion = fusion

    @torch.no_grad()
    def encode_task(self, scenario_goal: str, instruction: str, task: str):
        combined = f"{scenario_goal}. {instruction}. {task}"
        tokens = clip.tokenize([combined], truncate=True).to(self.device)
        return self.model.encode_text(tokens)  # [1, D]

    @torch.no_grad()
    def encode_objects(self, names: list[str], crops: list[Image.Image]):
        tokens = clip.tokenize(names).to(self.device)
        text_emb = self.model.encode_text(tokens)  # [N, D]

        images = torch.stack([self.preprocess(c) for c in crops]).to(self.device)
        vis_emb = self.model.encode_image(images)  # [N, D]

        if self.fusion == "mean":
            fused = (text_emb + vis_emb) / 2
        elif self.fusion == "text_only":
            fused = text_emb
        elif self.fusion == "vis_only":
            fused = vis_emb

        return fused  # [N, D]

    def score(self, scenario_goal: str, instruction: str, task: str,
              names: list[str], crops: list[Image.Image]):
        task_emb = self.encode_task(scenario_goal, instruction, task)  # [1, D]
        obj_emb  = self.encode_objects(names, crops)                   # [N, D]

        task_emb = task_emb / task_emb.norm(dim=-1, keepdim=True)
        obj_emb  = obj_emb  / obj_emb.norm(dim=-1, keepdim=True)

        scores = (obj_emb @ task_emb.T).squeeze(-1)  # [N]
        return scores


def main():
    scorer = RelevanceScorer(device="cpu", fusion="mean")

    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    names = ["window", "curtain", "radiator", "sofa", "lamp"]
    crops = [dummy] * 5

    scores = scorer.score(
        scenario_goal = "Ventilate the room",
        instruction   = "Open window near radiator and clear curtain above it",
        task          = "Rotate window handle to open",
        names         = names,
        crops         = crops,
    )

    for name, s in zip(names, scores):
        print(f"{name:12s}  {s:.4f}")


if __name__ == "__main__":
    main()