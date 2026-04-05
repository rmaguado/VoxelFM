import random
import numpy as np
import torch
from typing import Tuple
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import norm, t


def set_seed(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_cosine_scheduler_with_warmup(optimizer, cfg, epoch_len):
    cosine_iterations = cfg.cosine_epochs * epoch_len
    warmup_iterations = cfg.warmup_epochs * epoch_len

    def lr_lambda(step):
        if step < warmup_iterations:
            return (
                cfg.min_lr + (cfg.max_lr - cfg.min_lr) * step / warmup_iterations
            ) / cfg.max_lr

        elif step < warmup_iterations + cosine_iterations:
            t = step - warmup_iterations
            cosine = 0.5 * (
                1 + torch.cos(torch.tensor(t / cosine_iterations * torch.pi))
            )
            return (cfg.min_lr + (cfg.max_lr - cfg.min_lr) * cosine) / cfg.max_lr

        else:
            return cfg.min_lr / cfg.max_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def bootstrap_ci(
    errors: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean error metric.

    Args:
        errors: Array of errors (e.g., localization errors)
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound, standard_error)
    """
    n = len(errors)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_means.append(errors[indices].mean())

    bootstrap_means = np.array(bootstrap_means)
    se = np.std(bootstrap_means)
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return lower, upper, se


def roc_auc_ci_hanley(y_true, y_pred, alpha=0.05):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return 0.5, (0.5, 0.5)

    rocauc = float(roc_auc_score(y_true, y_pred))
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    Q1 = rocauc / (2 - rocauc)
    Q2 = 2 * rocauc**2 / (1 + rocauc)
    var = (
        rocauc * (1 - rocauc)
        + (n1 - 1) * (Q1 - rocauc**2)
        + (n0 - 1) * (Q2 - rocauc**2)
    ) / (n1 * n0)

    se = np.sqrt(var)
    crit = norm.ppf(1 - alpha / 2)
    ci = np.clip([rocauc - crit * se, rocauc + crit * se], 0, 1)

    return rocauc, tuple(ci)


def corrected_cv_ci_nadeau_bengio(k_rocauc, n_train, n_test, alpha=0.05):
    k_rocauc = np.array(k_rocauc)
    k = len(k_rocauc)
    mean_rocauc = np.mean(k_rocauc)
    variance = np.var(k_rocauc, ddof=1)
    correction = 1.0 + (n_test / n_train)
    corrected_var = variance * correction
    se_corrected = np.sqrt(corrected_var / k)
    crit = t.ppf(1 - alpha / 2, df=k - 1)
    ci_low = mean_rocauc - crit * se_corrected
    ci_high = mean_rocauc + crit * se_corrected
    return mean_rocauc, (max(0, ci_low), min(1, ci_high))


def compute_f1(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))
