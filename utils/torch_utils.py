import numpy as np
import torch
import random
import gc
import os

from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.cindex import ConcordanceIndex

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('high')

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def set_bn_eval(m):
    if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        m.eval()


def cox_loss_f64(log_hz, events, times, reduction="mean"):
    """Cox partial log-likelihood in float64: numerically unstable in float32"""
    with torch.amp.autocast("cuda", enabled=False):
        return neg_partial_log_likelihood(log_hz.double(), event=events, time=times, reduction=reduction)


def compute_survival_metrics(log_hz, events, times, new_time, with_loss=False, with_ci=False):
    """AUC and C-Index on a full set of predictions, optionally with Cox loss and bootstrap CIs"""
    auc_metric = Auc()
    cindex_metric = ConcordanceIndex()
    auc = auc_metric(log_hz, events, times, new_time=new_time)
    c_index = cindex_metric(log_hz, events, times)

    metrics = {"auc": auc.item(), "c_index": c_index.item()}

    if with_loss:
        metrics["loss"] = cox_loss_f64(log_hz, events, times).item()

    if with_ci:
        metrics["auc_ci"] = auc_metric.confidence_interval(method="bootstrap").tolist()
        metrics["c_index_ci"] = cindex_metric.confidence_interval(method="bootstrap").tolist()

    return metrics


def compute_group_grad_norms(optimizer):
    """L2 gradient norm of each optimizer param group"""
    norms = []
    for group in optimizer.param_groups:
        sq_sum = 0.0
        for p in group["params"]:
            if p.grad is not None:
                sq_sum += p.grad.data.norm(2).item() ** 2
        norms.append(sq_sum ** 0.5)
    return norms


def load_model_weights(model, checkpoint_path, device):
    """Load only the model weights from a checkpoint; raises on a missing path"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
