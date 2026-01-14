import argparse
from influence_functions import grad_z, inverse_hvp_lissa
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# define the model
class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.w(x.view(x.size(0), -1))

def main():
    parser = argparse.ArgumentParser(description="Influence Function Paper Reproduction")
    parser.add_argument("--n_train", type=int, default=12000)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--num_extremes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ihvp_t", type=int, default=5000)
    parser.add_argument("--ihvp_r", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cpu"
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Config: N={args.n_train}, L2={args.l2}, Extremes={args.num_extremes}")

    # Filter MNIST for Binary Classification
    tfm = transforms.ToTensor()
    dataset = datasets.MNIST("data", train=True, download=True, transform=tfm)
    
    idx = (dataset.targets == 0) | (dataset.targets == 1)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]
    
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=tfm)
    test_idx = (test_dataset.targets == 0) | (test_dataset.targets == 1)
    test_dataset.targets = test_dataset.targets[test_idx]
    test_dataset.data = test_dataset.data[test_idx]

    real_len = len(dataset)
    n_train = min(args.n_train, real_len)
    
    print(f"Dataset Size: {n_train}")
    train_ds = Subset(dataset, list(range(n_train)))
    
    all_loader = DataLoader(train_ds, batch_size=n_train, shuffle=False)
    x_train_all, y_train_all = next(iter(all_loader))
    x_train_all, y_train_all = x_train_all.to(device), y_train_all.to(device)

    # train the model with L-BFGS
    def train_with_lbfgs(model, x, y, l2_reg):
        optimizer = torch.optim.LBFGS(
            model.parameters(), 
            lr=1.0, 
            max_iter=100, 
            history_size=10, 
            line_search_fn="strong_wolfe"
        )
        def closure():
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            l2_loss = 0
            for p in model.parameters():
                l2_loss += 0.5 * l2_reg * (p ** 2).sum()
            total_loss = loss + l2_loss
            total_loss.backward()
            return total_loss
        model.train()
        optimizer.step(closure)
    
    model = LogisticRegression().to(device)
    train_with_lbfgs(model, x_train_all, y_train_all, args.l2)

    #choose the test points
    model.eval()
    z_test = None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    for i, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y).item()
        if loss > 0.3: 
            z_test = (x[0], y[0])
            break
    
    if z_test is None: z_test = (x[0], y[0])
    x_test, y_test = z_test
    
    def get_pure_loss(m):
        m.eval()
        with torch.no_grad():
            return F.cross_entropy(m(x_test.unsqueeze(0)), y_test.unsqueeze(0)).item()

    base_loss = get_pure_loss(model)
    print(f"Selected Test Point Loss: {base_loss:.4f}")

    #compute influence ussing Lissa
    v = grad_z(model, x_test, y_test)
    s_test = inverse_hvp_lissa(
        model, x_train_all, y_train_all, v, 
        l2=args.l2, damping=0.0, 
        t=args.ihvp_t, r=args.ihvp_r, scale=50.0
    )

    influences = []
    print(f"Calculating gradients for {n_train} points")
    for i in tqdm(range(n_train)):
        g_i = grad_z(model, x_train_all[i], y_train_all[i])
        inf = torch.dot(g_i, s_test).item() / n_train
        influences.append(inf)
    influences = np.array(influences)

    # Retraining with point removed
    K = args.num_extremes
    sorted_indices = np.argsort(influences) 
    neg_indices = sorted_indices[:K]
    pos_indices = sorted_indices[-K:]
    target_indices = np.concatenate([neg_indices, pos_indices])
    
    print(f"Selected {len(target_indices)} points.")

    pred_diffs = influences[target_indices]
    actual_diffs = []
    
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    for idx in tqdm(target_indices):
        mask = torch.ones(n_train, dtype=torch.bool)
        mask[idx] = False
        x_loo = x_train_all[mask]
        y_loo = y_train_all[mask]

        m_loo = LogisticRegression().to(device)
        m_loo.load_state_dict(base_state)
        train_with_lbfgs(m_loo, x_loo, y_loo, args.l2)
        
        loss_new = get_pure_loss(m_loo)
        actual_diffs.append(loss_new - base_loss)

    # get th final picture
    plt.style.use('seaborn-v0_8-whitegrid')
  
    plt.figure(figsize=(5, 5))
  
    max_val = max(np.max(np.abs(pred_diffs)), np.max(np.abs(actual_diffs)))
    if max_val == 0: max_val = 0.1
    mval = max_val * 1.1
    
    plt.plot([-mval, mval], [-mval, mval], color='gray', linestyle='--', alpha=0.5, linewidth=1.5)#diagonal
    
    plt.scatter(actual_diffs, pred_diffs, alpha=0.8, color="#4c72b0", s=40, edgecolors='white', linewidth=0.5)#points
    
    plt.xlabel('Actual change in loss', fontsize=12)#axis
    plt.ylabel('Predicted change in loss', fontsize=12)
        
    plt.xlim(-mval, mval)# same range
    plt.ylim(-mval, mval)
    
    plt.tight_layout()
    
    out_path = os.path.join("outputs", f"Linear (approx).png")
    plt.savefig(out_path, dpi=300)

if __name__ == "__main__":

    main()


