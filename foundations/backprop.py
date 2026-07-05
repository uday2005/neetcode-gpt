import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self , x):
        output = 1 / (1+ np.exp(-x))
        return output 

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x , w) + b
        y_hat = self.sigmoid(z)
        loss = 0.5 * (np.square(y_hat - y_true))
        dl_db = -1 * y_hat * (y_true - y_hat) * (1 - y_hat)
        dl_dw = -1 * y_hat * (y_true - y_hat) * (1 - y_hat) * x

        return (np.round(dl_dw , 5) , np.round(dl_db , 5))





