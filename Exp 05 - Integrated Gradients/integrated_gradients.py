def integrated_gradients(model, x, baseline, target, steps=50):
    total_grad = torch.zeros_like(x)
    for k in range(1, steps + 1):
        point = baseline + (k / steps) * (x - baseline)     # a point on the straight path
        point = point.clone().requires_grad_(True)
        score = model(point)[0, target]                     # class score (logit, before softmax)
        grad = torch.autograd.grad(score, point)[0]         # d score / d pixels
        total_grad += grad
    avg_grad = total_grad / steps                           # Riemann-sum average of the gradients
    return (x - baseline) * avg_grad                        # scale by how far the input moved
