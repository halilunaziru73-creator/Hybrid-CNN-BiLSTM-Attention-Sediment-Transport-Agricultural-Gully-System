"""
Hybrid CNN-BiLSTM-Attention model (paper Section 2.7.1) assembled from the
layers in layers.py, plus Adam (Eq. 27) and the weighted Huber training
loss (Eq. 8i) with L2 decay (Eq. 21).

Architecture (Section 2.14.1's retained configuration):
  Conv1D(1->8, k=3, ReLU, same) -> Conv1D(8->16, k=3, ReLU, same)
  -> BiLSTM(16->8 per direction, 16 total)
  -> Attention pooling (key dim 8)
  -> Dense(8->16, ReLU) -> Dropout(p=0.2)
  -> Dense(16->1, linear)

Input: 6 standardised predictors treated as a length-T=6, 1-channel
sequence (each "timestep" is one physical variable in a fixed order).

Trained sample-by-sample (the paper does not specify a mini-batch size;
at n<=100 a per-sample update is a defensible, explicit choice) with
Adam, weighted Huber loss and L2 weight decay.
"""
import numpy as np
from .layers import Conv1D, BiLSTM, AttentionPool, Dense, Dropout, huber_loss, l2_grad


class HybridCNNBiLSTMAttention:
    def __init__(self, seed=42, conv_filters=(8, 16), lstm_units=8, attn_dim=8,
                 dense_units=16, dropout_p=0.2, l2_alpha=1e-3, huber_delta=1.0,
                 lr=0.01):
        self.rng = np.random.default_rng(seed)
        self.conv1 = Conv1D(1, conv_filters[0], k=3, rng=self.rng)
        self.conv2 = Conv1D(conv_filters[0], conv_filters[1], k=3, rng=self.rng)
        self.bilstm = BiLSTM(conv_filters[1], lstm_units, rng=self.rng)
        self.attn = AttentionPool(2 * lstm_units, attn_dim, rng=self.rng)
        self.dense1 = Dense(attn_dim, dense_units, rng=self.rng, relu=True)
        self.dropout = Dropout(dropout_p)
        self.dense2 = Dense(dense_units, 1, rng=self.rng, relu=False)
        self.l2_alpha = l2_alpha
        self.huber_delta = huber_delta
        self.lr = lr

        self._param_map = self._collect_params()
        self._adam_m = {k: np.zeros_like(v) for k, v in self._param_map.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self._param_map.items()}
        self._adam_t = 0

    # -- parameter bookkeeping -------------------------------------------------
    def _collect_params(self):
        p = {}
        p["conv1.W"], p["conv1.b"] = self.conv1.W, self.conv1.b
        p["conv2.W"], p["conv2.b"] = self.conv2.W, self.conv2.b
        p["bilstm.fwd.Wx"], p["bilstm.fwd.Wh"], p["bilstm.fwd.b"] = (
            self.bilstm.fwd.Wx, self.bilstm.fwd.Wh, self.bilstm.fwd.b)
        p["bilstm.bwd.Wx"], p["bilstm.bwd.Wh"], p["bilstm.bwd.b"] = (
            self.bilstm.bwd.Wx, self.bilstm.bwd.Wh, self.bilstm.bwd.b)
        p["attn.q"], p["attn.K"], p["attn.V"] = self.attn.q, self.attn.K, self.attn.V
        p["dense1.W"], p["dense1.b"] = self.dense1.W, self.dense1.b
        p["dense2.W"], p["dense2.b"] = self.dense2.W, self.dense2.b
        return p

    def get_flat_params(self):
        return {k: v.copy() for k, v in self._param_map.items()}

    def set_flat_params(self, params):
        for k, v in params.items():
            self._param_map[k][...] = v

    # -- forward / backward for ONE sample -------------------------------------
    def forward(self, x, training, dropout_mask=None):
        """x: shape (6,) standardised feature vector."""
        x_seq = x.reshape(-1, 1)  # (T=6, c_in=1)
        a1, cache1 = self.conv1.forward(x_seq)
        a2, cache2 = self.conv2.forward(a1)
        H_bi, cache_bilstm = self.bilstm.forward(a2)
        c, alpha, cache_attn = self.attn.forward(H_bi)
        a_dense1, cache_dense1 = self.dense1.forward(c)
        a_drop, mask = self.dropout.forward(a_dense1, training, self.rng, mask=dropout_mask)
        yhat, cache_dense2 = self.dense2.forward(a_drop)
        yhat = yhat[0]
        cache = dict(cache1=cache1, cache2=cache2, cache_bilstm=cache_bilstm,
                     cache_attn=cache_attn, cache_dense1=cache_dense1,
                     mask=mask, cache_dense2=cache_dense2, alpha=alpha)
        return yhat, cache

    def backward(self, dyhat, cache):
        d_adrop, g_dense2 = self.dense2.backward(np.array([dyhat]), cache["cache_dense2"])
        d_adense1 = self.dropout.backward(d_adrop, cache["mask"])
        dc, g_dense1 = self.dense1.backward(d_adense1, cache["cache_dense1"])
        dH_bi, g_attn = self.attn.backward(dc, cache["cache_attn"])
        da2, g_bilstm = self.bilstm.backward(dH_bi, cache["cache_bilstm"])
        da1, g_conv2 = self.conv2.backward(da2, cache["cache2"])
        _, g_conv1 = self.conv1.backward(da1, cache["cache1"])

        grads = {}
        grads["conv1.W"], grads["conv1.b"] = g_conv1["W"], g_conv1["b"]
        grads["conv2.W"], grads["conv2.b"] = g_conv2["W"], g_conv2["b"]
        grads["bilstm.fwd.Wx"] = g_bilstm["fwd"]["Wx"]
        grads["bilstm.fwd.Wh"] = g_bilstm["fwd"]["Wh"]
        grads["bilstm.fwd.b"] = g_bilstm["fwd"]["b"]
        grads["bilstm.bwd.Wx"] = g_bilstm["bwd"]["Wx"]
        grads["bilstm.bwd.Wh"] = g_bilstm["bwd"]["Wh"]
        grads["bilstm.bwd.b"] = g_bilstm["bwd"]["b"]
        grads["attn.q"], grads["attn.K"], grads["attn.V"] = g_attn["q"], g_attn["K"], g_attn["V"]
        grads["dense1.W"], grads["dense1.b"] = g_dense1["W"], g_dense1["b"]
        grads["dense2.W"], grads["dense2.b"] = g_dense2["W"], g_dense2["b"]
        return grads

    def loss_and_grads(self, x, y, training=True):
        yhat, cache = self.forward(x, training=training)
        loss, dloss_dyhat = huber_loss(y, yhat, delta=self.huber_delta)
        l2 = sum(np.sum(p ** 2) for p in self._param_map.values())
        loss = loss + self.l2_alpha * l2
        grads = self.backward(dloss_dyhat, cache)
        for k in grads:
            grads[k] = grads[k] + l2_grad(self._param_map[k], self.l2_alpha)
        return loss, grads, yhat

    # -- Adam step ---------------------------------------------------------
    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._adam_t += 1
        t = self._adam_t
        for k, g in grads.items():
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * (g ** 2)
            m_hat = self._adam_m[k] / (1 - beta1 ** t)
            v_hat = self._adam_v[k] / (1 - beta2 ** t)
            self._param_map[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    def n_params(self):
        return sum(v.size for v in self._param_map.values())

    def predict(self, X, training=False, dropout_mask=None):
        preds = np.zeros(len(X))
        for i, x in enumerate(X):
            preds[i], _ = self.forward(x, training=training, dropout_mask=dropout_mask)
        return preds
