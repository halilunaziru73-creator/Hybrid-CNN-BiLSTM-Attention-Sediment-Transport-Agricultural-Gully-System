"""
Hand-rolled NumPy layers for the Hybrid CNN-BiLSTM-Attention model
(paper Section 2.7.1, Equations 8a-8h).

No autodiff framework is available in this environment (matching the
paper's own claim of a from-scratch NumPy implementation with no GPU
framework), so every layer below implements both a forward pass and an
explicit analytic backward pass. `gradcheck.py` verifies every gradient
here against finite differences before any of it is trusted for training,
exactly as Section 2.7.1 describes ("every analytic gradient verified
against finite differences, max relative error < 1e-9").

Shapes follow the paper: each sample is a length T=6 sequence (one
"position" per input variable) with 1 input channel.
"""
import numpy as np


def glorot(shape, rng):
    """Glorot/Xavier uniform init, Var(w) = 2/(n_in + n_out)  (Eq. 24)."""
    if len(shape) == 2:
        n_in, n_out = shape
    elif len(shape) == 3:  # conv kernel (k, c_in, c_out)
        n_in, n_out = shape[0] * shape[1], shape[2]
    else:
        raise ValueError(shape)
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, size=shape)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# --------------------------------------------------------------------------
# Conv1D, kernel k, "same" padding, ReLU activation fused in (Eq. 8a)
# --------------------------------------------------------------------------
class Conv1D:
    def __init__(self, c_in, c_out, k, rng):
        self.k, self.c_in, self.c_out = k, c_in, c_out
        self.W = glorot((k, c_in, c_out), rng)
        self.b = np.zeros(c_out)
        self.pad = k // 2

    def params(self):
        return {"W": self.W, "b": self.b}

    def forward(self, x):
        # x: (T, c_in) -> pre-activation z: (T, c_out) -> ReLU
        T = x.shape[0]
        xp = np.pad(x, ((self.pad, self.pad), (0, 0)))
        z = np.zeros((T, self.c_out))
        for i in range(T):
            window = xp[i:i + self.k]                    # (k, c_in)
            z[i] = np.tensordot(window, self.W, axes=([0, 1], [0, 1])) + self.b
        a = np.maximum(z, 0.0)
        cache = (x, xp, z)
        return a, cache

    def backward(self, dA, cache):
        x, xp, z = cache
        T = x.shape[0]
        dZ = dA * (z > 0)
        dW = np.zeros_like(self.W)
        db = dZ.sum(axis=0)
        dXp = np.zeros_like(xp)
        for i in range(T):
            window = xp[i:i + self.k]
            dW += np.einsum("kc,o->kco", window, dZ[i])
            dXp[i:i + self.k] += np.einsum("o,kco->kc", dZ[i], self.W)
        dX = dXp[self.pad:self.pad + T] if self.pad else dXp
        return dX, {"W": dW, "b": db}


# --------------------------------------------------------------------------
# Single-direction LSTM cell run over a sequence, full BPTT (Eq. 8b-8c)
# --------------------------------------------------------------------------
class LSTMDirection:
    def __init__(self, c_in, h, rng, reverse=False):
        self.c_in, self.h, self.reverse = c_in, h, reverse
        # stacked gates order: i, f, o, g
        self.Wx = glorot((c_in, 4 * h), rng)
        self.Wh = glorot((h, 4 * h), rng)
        self.b = np.zeros(4 * h)
        self.b[h:2 * h] = 1.0  # forget-gate bias init to 1 (Jozefowicz et al., 2015)

    def params(self):
        return {"Wx": self.Wx, "Wh": self.Wh, "b": self.b}

    def forward(self, x):
        # x: (T, c_in)
        T, H = x.shape[0], self.h
        order = range(T - 1, -1, -1) if self.reverse else range(T)
        order = list(order)
        h_prev = np.zeros(H)
        c_prev = np.zeros(H)
        H_out = np.zeros((T, H))
        cache_steps = []
        for t in order:
            gates = x[t] @ self.Wx + h_prev @ self.Wh + self.b
            i = sigmoid(gates[0:H])
            f = sigmoid(gates[H:2 * H])
            o = sigmoid(gates[2 * H:3 * H])
            g = np.tanh(gates[3 * H:4 * H])
            c = f * c_prev + i * g
            h = o * np.tanh(c)
            cache_steps.append((x[t], h_prev, c_prev, i, f, o, g, c, h))
            H_out[t] = h
            h_prev, c_prev = h, c
        return H_out, {"steps": cache_steps, "order": order, "T": T}

    def backward(self, dH_out, cache):
        H = self.h
        steps, order, T = cache["steps"], cache["order"], cache["T"]
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        db = np.zeros_like(self.b)
        dX = np.zeros((T, self.c_in))
        dh_next = np.zeros(H)
        dc_next = np.zeros(H)
        for idx in range(T - 1, -1, -1):
            t = order[idx]
            x_t, h_prev, c_prev, i, f, o, g, c, h = steps[idx]
            dh = dH_out[t] + dh_next
            do = dh * np.tanh(c)
            dc = dh * o * (1 - np.tanh(c) ** 2) + dc_next
            di = dc * g
            df = dc * c_prev
            dg = dc * i
            dc_prev = dc * f

            d_i_pre = di * i * (1 - i)
            d_f_pre = df * f * (1 - f)
            d_o_pre = do * o * (1 - o)
            d_g_pre = dg * (1 - g ** 2)
            dgates = np.concatenate([d_i_pre, d_f_pre, d_o_pre, d_g_pre])

            dWx += np.outer(x_t, dgates)
            dWh += np.outer(h_prev, dgates)
            db += dgates
            dX[t] = dgates @ self.Wx.T
            dh_next = dgates @ self.Wh.T
            dc_next = dc_prev
        return dX, {"Wx": dWx, "Wh": dWh, "b": db}


class BiLSTM:
    """Forward + reverse LSTM, hidden states concatenated per position (Eq. 8d)."""

    def __init__(self, c_in, h, rng):
        self.fwd = LSTMDirection(c_in, h, rng, reverse=False)
        self.bwd = LSTMDirection(c_in, h, rng, reverse=True)
        self.h = h

    def params(self):
        return {"fwd": self.fwd.params(), "bwd": self.bwd.params()}

    def forward(self, x):
        Hf, cache_f = self.fwd.forward(x)
        Hb, cache_b = self.bwd.forward(x)
        H_bi = np.concatenate([Hf, Hb], axis=1)  # (T, 2h)
        return H_bi, {"f": cache_f, "b": cache_b}

    def backward(self, dH_bi, cache):
        H = self.h
        dHf, dHb = dH_bi[:, :H], dH_bi[:, H:]
        dXf, gf = self.fwd.backward(dHf, cache["f"])
        dXb, gb = self.bwd.backward(dHb, cache["b"])
        return dXf + dXb, {"fwd": gf, "bwd": gb}


# --------------------------------------------------------------------------
# Scaled dot-product attention pooling with a single learned query (Eq. 8e-g)
# --------------------------------------------------------------------------
class AttentionPool:
    def __init__(self, d_in, d_k, rng):
        self.q = glorot((d_k,), rng) if False else rng.normal(0, 1.0 / np.sqrt(d_k), size=d_k)
        self.K = glorot((d_in, d_k), rng)
        self.V = glorot((d_in, d_k), rng)
        self.d_k = d_k

    def params(self):
        return {"q": self.q, "K": self.K, "V": self.V}

    def forward(self, H_bi):
        # H_bi: (T, d_in)
        Kp = H_bi @ self.K                     # (T, d_k)
        Vp = H_bi @ self.V                     # (T, d_k)
        e = (Kp @ self.q) / np.sqrt(self.d_k)  # (T,)
        e_shift = e - e.max()
        exp_e = np.exp(e_shift)
        alpha = exp_e / exp_e.sum()            # (T,)
        c = alpha @ Vp                         # (d_k,)
        cache = (H_bi, Kp, Vp, alpha)
        return c, alpha, cache

    def backward(self, dc, cache):
        H_bi, Kp, Vp, alpha = cache
        T, d_k = Kp.shape
        # dC/dVp and dC/dalpha
        dVp = np.outer(alpha, dc)                       # (T, d_k)
        dalpha = Vp @ dc                                 # (T,)
        # softmax backward: dalpha -> de
        de = alpha * (dalpha - (alpha @ dalpha))
        # de/dKp via e = Kp @ q / sqrt(dk)
        dKp = np.outer(de / np.sqrt(d_k), self.q)        # (T, d_k)
        dq = (Kp.T @ (de / np.sqrt(d_k)))                # (d_k,)
        dK = H_bi.T @ dKp
        dV = H_bi.T @ dVp
        dH_bi = dKp @ self.K.T + dVp @ self.V.T
        return dH_bi, {"q": dq, "K": dK, "V": dV}


# --------------------------------------------------------------------------
# Dense (+ optional ReLU) and Dropout
# --------------------------------------------------------------------------
class Dense:
    def __init__(self, n_in, n_out, rng, relu=True):
        self.W = glorot((n_in, n_out), rng)
        self.b = np.zeros(n_out)
        self.relu = relu

    def params(self):
        return {"W": self.W, "b": self.b}

    def forward(self, x):
        z = x @ self.W + self.b
        a = np.maximum(z, 0.0) if self.relu else z
        return a, (x, z)

    def backward(self, da, cache):
        x, z = cache
        dz = da * (z > 0) if self.relu else da
        dW = np.outer(x, dz)
        db = dz
        dx = dz @ self.W.T
        return dx, {"W": dW, "b": db}


class Dropout:
    def __init__(self, p):
        self.p = p

    def forward(self, x, training, rng, mask=None):
        if not training and mask is None:
            return x, np.ones_like(x)
        if mask is None:
            mask = (rng.uniform(size=x.shape) > self.p).astype(x.dtype) / (1 - self.p)
        return x * mask, mask

    def backward(self, dout, mask):
        return dout * mask


# --------------------------------------------------------------------------
# Weighted Huber loss (Eq. 8i) and L2 penalty (Eq. 21)
# --------------------------------------------------------------------------
def huber_loss(y, yhat, delta=1.0):
    e = y - yhat
    if abs(e) <= delta:
        loss = 0.5 * e ** 2
        dloss_dyhat = -e
    else:
        loss = delta * (abs(e) - 0.5 * delta)
        dloss_dyhat = -delta * np.sign(e)
    return loss, dloss_dyhat


def l2_penalty(param_list, alpha):
    return alpha * sum(np.sum(p ** 2) for p in param_list)


def l2_grad(param, alpha):
    return 2 * alpha * param
