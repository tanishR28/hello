def grad_x_input(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    mask = (z1 > 0).astype(float)                      # 1 where ReLU is active, else 0
    gradient = W1 @ (mask * W2[:, 0])                  # d(logit)/dx  by the chain rule
    return x * gradient                                # gradient x input
