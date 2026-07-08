import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]], b1: List[float],
        W2: List[List[float]], b2: List[float],
        y_true: List[float]
    ) -> dict:

        # Convert everything to numpy arrays
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # ---------- Forward Pass ----------
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)           # ReLU

        y_hat = W2 @ a1 + b2

        loss = np.mean((y_hat - y_true) ** 2)

        # ---------- Backward Pass ----------

        # dL/dy_hat
        dL_dy = 2 * (y_hat - y_true) / len(y_true)

        # Output layer
        dW2 = np.outer(dL_dy, a1)
        db2 = dL_dy

        # Gradient flowing into hidden activation
        dL_da1 = W2.T @ dL_dy

        # ReLU derivative
        relu_grad = (z1 > 0).astype(float)

        dL_dz1 = dL_da1 * relu_grad

        # First layer
        dW1 = np.outer(dL_dz1, x)
        db1 = dL_dz1

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }