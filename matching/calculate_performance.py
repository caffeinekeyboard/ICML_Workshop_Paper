import torch
from pathlib import Path
from matcher import matcher
from matching_metrics import hybrid_metric
from matching_performance import fcv_frr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERMEDIATES_PATH = PROJECT_ROOT / "intermediates_batch_0.pt"

data = matcher.load_intermediates(str(INTERMEDIATES_PATH))
alignment1 = data["alignment1"]
original_img = data["origninal_img"]
features1 = data["extractor1"]
features2 = data["extractor2"]

scores = hybrid_metric(alignment1, original_img, features1, features2, alpha=0.45, beta=0.45, delta=0.1, mask=False)
performance = fcv_frr(scores)

def _avg_minutiae_count(features):
	if isinstance(features, (list, tuple)):
		counts = [f.shape[0] for f in features]
		return sum(counts) / len(counts) if counts else 0.0
	if torch.is_tensor(features):
		return float(features.shape[0])
	return 0.0

avg_minutiae_f1 = _avg_minutiae_count(features1)
avg_minutiae_f2 = _avg_minutiae_count(features2)

score_tensor = scores if torch.is_tensor(scores) else torch.tensor(scores)
score_min = float(score_tensor.min())
score_max = float(score_tensor.max())
score_var = float(score_tensor.var(unbiased=False))

print(f"Avg minutiae count (features1): {avg_minutiae_f1:.4f}")
print(f"Avg minutiae count (features2): {avg_minutiae_f2:.4f}")
print(f"Score matrix min: {score_min:.6f}")
print(f"Score matrix max: {score_max:.6f}")
print(f"Score matrix var: {score_var:.6f}")

print(performance)