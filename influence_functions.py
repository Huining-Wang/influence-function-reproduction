import torch
import torch.nn.functional as F

def flatten(tensors):
    return torch.cat([t.flatten() for t in tensors])

#pure loss (without regularization)
def loss_z(model, x, y):
    logits = model(x.unsqueeze(0)) #the batch dimension
    loss = F.cross_entropy(logits, y.unsqueeze(0), reduction="sum")
    return loss

#gradient of the loss w.r.t parameters
def grad_z(model, x, y):
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_z(model, x, y)
    grads = torch.autograd.grad(loss, params)
    return flatten([g.detach() for g in grads])

#Hessian-Vector Product: H*v
def hvp_z(model, x, y, v, l2=0.0):
    #Includes the L2 regularization term in the curvature: (H_loss + l2*I) * v
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    # 1. Compute Gradient of Loss
    loss = loss_z(model, x, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_vec = flatten(grads)
    
    # 2. Compute Gradient of (Gradient * v) -> H*v
    gv = (g_vec * v).sum()
    hv = torch.autograd.grad(gv, params)
    hv_vec = flatten(hv).detach() # stop graph growth in LiSSA
    
    # 3. Add L2 regularization term: H_total = H_loss + l2*I
    # (H_loss + l2*I) * v = H_loss*v + l2*v
    if l2 > 0:
        hv_vec += l2 * v
        
    return hv_vec

#Estimates (H + damping*I)^(-1) * v using LiSSA
def inverse_hvp_lissa(
    model,
    x_train,
    y_train,
    v,
    l2=0.0,
    t=5000,
    r=10,
    scale=25.0,
    damping=0.01,
    generator=None,
):
    n = x_train.shape[0]
    v = v.detach() 

    # Recursion: s_j = v + (I - H) * s_{j-1}
    # scaling: s_j = v + (I - H/scale) * s_{j-1}
    # approximates H^{-1} * v / scale
    
    def one_run():
        s = v.clone()
        for _ in range(t):
            # Sample a single training point
            i = torch.randint(0, n, (1,), generator=generator).item()
            
            # Compute HVP on this point
            hv = hvp_z(model, x_train[i], y_train[i], s, l2)
            
            # Update estimate
            # (1 - damping)  to improve stability for non-convex objectives
            s = v + (1 - damping) * s - hv / scale
            
        return s / scale # unscaled inverse estimate

    # reduce variance
    return torch.stack([one_run() for _ in range(r)]).mean(0)