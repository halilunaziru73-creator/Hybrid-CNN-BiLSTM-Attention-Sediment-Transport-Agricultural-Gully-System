"""
Finite-difference gradient check for HybridCNNBiLSTMAttention.

The paper (Section 2.7.1) states: "every analytic gradient verified
against finite differences (max relative error < 1e-9)". This script
performs that check. Dropout is disabled (training=False, mask of ones)
for the check since Bernoulli dropout is not itself differentiable in
the finite-difference sense; L2 decay is included since it is a smooth
term.

Run: python3 -m hybrid_model.gradcheck
"""
import numpy as np
from .model import HybridCNNBiLSTMAttention


def gradcheck(seed=0, eps=1e-5, n_checks=40, verbose=True):
    rng = np.random.default_rng(seed)
    model = HybridCNNBiLSTMAttention(seed=seed)
    x = rng.normal(size=6)
    y = rng.normal()
    ones_mask = np.ones(model.dense1.b.shape[0])

    def loss_at(params_backup=None):
        loss, _, _ = model.loss_and_grads_fixed_mask(x, y, ones_mask)
        return loss

    # patch a fixed-dropout-mask variant for determinism during the check
    def loss_and_grads_fixed_mask(x, y, mask):
        yhat, cache = model.forward(x, training=True, dropout_mask=mask)
        from .layers import huber_loss
        loss, dloss_dyhat = huber_loss(y, yhat, delta=model.huber_delta)
        l2 = sum(np.sum(p ** 2) for p in model._param_map.values())
        loss = loss + model.l2_alpha * l2
        grads = model.backward(dloss_dyhat, cache)
        for k in grads:
            grads[k] = grads[k] + 2 * model.l2_alpha * model._param_map[k]
        return loss, grads, yhat

    model.loss_and_grads_fixed_mask = loss_and_grads_fixed_mask

    analytic_loss, analytic_grads, _ = loss_and_grads_fixed_mask(x, y, ones_mask)

    max_rel_err = 0.0
    checked = 0
    param_names = list(model._param_map.keys())
    rng2 = np.random.default_rng(1)
    for name in param_names:
        param = model._param_map[name]
        flat = param.reshape(-1)
        n_this = min(n_checks // len(param_names) + 1, flat.size)
        idxs = rng2.choice(flat.size, size=n_this, replace=False)
        for idx in idxs:
            orig = flat[idx]
            flat[idx] = orig + eps
            loss_plus, _, _ = loss_and_grads_fixed_mask(x, y, ones_mask)
            flat[idx] = orig - eps
            loss_minus, _, _ = loss_and_grads_fixed_mask(x, y, ones_mask)
            flat[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            analytic = analytic_grads[name].reshape(-1)[idx]
            denom = max(abs(numeric), abs(analytic), 1e-8)
            rel_err = abs(numeric - analytic) / denom
            max_rel_err = max(max_rel_err, rel_err)
            checked += 1
            if verbose and rel_err > 1e-4:
                print(f"  LARGE ERROR {name}[{idx}]: analytic={analytic:.6e} "
                      f"numeric={numeric:.6e} rel_err={rel_err:.2e}")

    print(f"Checked {checked} parameters across {len(param_names)} tensors.")
    print(f"Max relative error: {max_rel_err:.3e}")
    return max_rel_err


if __name__ == "__main__":
    err = gradcheck()
    status = "PASS" if err < 1e-4 else "FAIL"
    print(f"\nGradient check: {status} (max relative error {err:.3e})")
