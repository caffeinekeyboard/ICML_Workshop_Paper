import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
	sys.path.insert(0, _PROJECT_ROOT)

from datasets.no_split_dataloader import get_no_split_dataloader
from matching.matcher import matcher
from matching.matching_performance import fcv_frr
from matching_metrics import hybrid_metric, minutiae_metric
from matching.fingernet_wrapper import FingerNetWrapper
from matching.fingernet import FingerNet
from model.gumnet import GumNet
from datasets.eval_loader_data import EvalDataset, get_eval_dataloader
from datasets.original_data import OrigDataset, get_orig_dataloader
from nbis_extractor import Nbis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = PROJECT_ROOT / "model" / "gumnet_2d_best_noise_level_0_8x8_200.pth"
FN_WEIGHTS_PATH = PROJECT_ROOT / "matching" / "fingernet.pth"

class IdentityAlignment(nn.Module):
	"""
	Alignment stub that returns the first batch unchanged.
	"""
	def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
		return img1

def gumnet_init(device: torch.device):
	model = GumNet(grid_size=8)
	model.load_state_dict(WEIGHTS_PATH, map_location=device, strict=True)
	return model

def fingernet_init():
	fingernet = FingerNet()
	fingernet.load_state_dict(torch.load(FN_WEIGHTS_PATH, map_location='cpu'))
	return fingernet

def build_matcher(device: torch.device, use_mask: bool = False) -> matcher:
	alignment = gumnet_init(device)
	print("1/4 Gumnet built.")
	#fingernet = fingernet_init()
	#print("2/4 FingerNet built.")
	#extractor = FingerNetWrapper(fingernet, minutiae_threshold=0.1, max_candidates=100)
	#print("3/4 FingerNet wrapper built.")
	extractor = Nbis()
	print("2+3/4 Nbis extractor built.")
	model = matcher(alignment, extractor, hybrid_metric, fcv_frr, mask=use_mask, gumnet=True)
	print("4/4 Matcher built.")
	model.to(device)
	model.eval()
	return model

def run_inference(
	data_root: str,
	batch_size: int = (8*7) // 2,
	num_workers: int = 0,
	max_batches: Optional[int] = None,
) -> None:
	# Limit CPU thread usage to reduce potential stalling in feature extraction
	try:
		torch.set_num_threads(1)
		torch.set_num_interop_threads(1)
	except Exception:
		pass
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	print("Building matcher model...")
	model = build_matcher(device=device)
	
	print("Creating dataloaders...")
	loader1 = get_eval_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)
	loader2 = get_orig_dataloader(
		data_root=data_root,
		batch_size=batch_size,
		num_workers=num_workers,
	)

	print("Running inference...")
	all_scores = []
	with torch.inference_mode():
		for batch_idx, (batch1, batch2) in enumerate(zip(loader1, loader2)):
			if batch_idx>0:
				break
			Sa = batch1["Sa"].to(device)
			Sb = batch1["Sb"].to(device)
			orig_Sa = batch2["Sa"].to(device)
			orig_Sb = batch2["Sb"].to(device)
			scores = model(Sa, Sb, orig_Sa, orig_Sb)
			model.save_intermediates(f"intermediates_batch_{batch_idx}.pt")
			if not isinstance(scores, torch.Tensor):
				scores = torch.tensor(scores, dtype=torch.float32)
			all_scores.append(scores.detach().cpu())

			if max_batches is not None and (batch_idx + 1) >= max_batches:
				break
	
	print("Inference completed.")
	print("Calculating performance...")
	if all_scores:
		scores_tensor = torch.cat([
			s.view(-1) if isinstance(s, torch.Tensor) else torch.tensor([s], dtype=torch.float32)
			for s in all_scores
		], dim=0)
		print(f"Processed {scores_tensor.numel()} samples.")
		print(f"Performance: {scores_tensor.mean().item():.4f}")


if __name__ == "__main__":
	_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
	DATA_ROOT = os.path.abspath(
		os.path.join(_SCRIPT_DIR, "..", "data", "FCV", "FVC2004", "Dbs", "DB1_B")
	)

	run_inference(
		data_root=DATA_ROOT,
		max_batches=None,
	)
