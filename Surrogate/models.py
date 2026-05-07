import torch
import torch.nn as nn

import numpy as np
from scipy.integrate import solve_ivp
from torch.xpu import device
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy

# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def wrap(angle):
    """Wrap angle(s) to [-π, π]."""
    #print(angle)
    return (angle + np.pi) % (2 * np.pi) - np.pi

"""def preprocess(vector):
    Wrap a numpy angle vector and convert to float32 tensor.
    wrapped = wrap(vector)
    tensor = torch.tensor(wrapped, dtype=torch.float32)
    return tensor"""

# ──────────────────────────────────────────────
# Example Models
# Structure 4D vector in -> Model-> 4D vector output
# ──────────────────────────────────────────────
class PendulumSurrogate(nn.Module):
    """4-layer MLP modified"""
    # Layers could be varied to see the influence
    def __init__(self, input_shape=4,output_shape=4,hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_shape, hidden_size),  # Input: [theta, omega, dt]
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_shape)  # Output: [theta_next, omega_next]
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, n=3, latent_dim: int = 2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(16 * n, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n)  -->  unsqueeze to (batch, 1, n)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv_layers(x)       # (batch, 16, n)
        x = x.view(x.size(0), -1)     # (batch, 16*n)
        x = self.fc(x)                # (batch, latent_dim)
        return x


class Decoder(nn.Module):
    def __init__(self, n_in_out=4, latent_dim: int = 2):
        super().__init__()
        self.n_in_out = n_in_out
        self.fc = nn.Linear(latent_dim, 16 * n_in_out)
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)                        # (batch, 16*n_in_out)
        x = x.view(x.size(0), 16, self.n_in_out)  # (batch, 16, n_in_out)
        x = self.conv_layers(x)               # (batch, 1, n_in_out)
        x = x.squeeze(1)                      # (batch, n_in_out)  <-- key fix
        return x


class CNNAutoencoder(nn.Module):
    def __init__(self, n_in=4, n_out=4, latent_dim: int = 8):
        super().__init__()
        self.encoder = Encoder(n=n_in, latent_dim=latent_dim)
        self.decoder = Decoder(n_in_out=n_out, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

class DeepFeedForwardNet(nn.Module):
    """My new model 1"""
    def __init__(self, input_shape=4, output_shape=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, output_shape)
        )

    def forward(self, x):
        return self.net(x)
    

def make_surrogate_dynamics(model):
    model.eval()
    device=next(model.parameters()).device

    def dynamics(t, y):
        theta = wrap(y[0])
        dtheta = y[1]  # not an angle, no wrap
        omega = wrap(y[2])
        domega = y[3]  # not an angle, no wrap

        x = torch.tensor([[theta, dtheta, omega, domega]],
                         dtype=torch.float32).to(device)  # GPU support

        with torch.no_grad():  # no grad graph
            dydt = model(x).cpu().numpy()[0]  # back to CPU numpy

        return dydt  # [dtheta, ddtheta, domega, ddomega]

    return dynamics
def run_model_surrogate_dynamics_solve_ip(x0,model,t_span=(0,10),step_size=1e-2):
    t_eval = np.linspace(*t_span, int(t_span[1]/step_size))
    print(t_eval)
    surrogate_dynamics = make_surrogate_dynamics(model)
    print(surrogate_dynamics)
    sol_surrogate = solve_ivp(
        fun=surrogate_dynamics,
        t_span=t_span,
        y0=x0,
        t_eval=t_eval,
        method='Radau'
    )
    print(sol_surrogate)
    return sol_surrogate.y.T

def solve_pendulum_surrogate(model, y0, t_span, t_eval):
    """Solve pendulum using trained surrogate model"""
    device = next(model.parameters()).device
    model.eval()

    trajectory = [y0]
    current_state = np.array(y0)

    with torch.no_grad():
        for i in range(len(t_eval) - 1):

            x_input = torch.FloatTensor(np.concatenate([current_state])).unsqueeze(0).to(device)
            print(x_input)
            next_state = model(x_input).cpu().numpy()[0]
            trajectory.append(next_state)
            current_state = next_state

    return np.array(trajectory)

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_surrogate_simple(X_train, Y_train,model_=PendulumSurrogate, epochs=1000, lr=0.001, batch_size=128):
    """Train the surrogate model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train).to(device)
    Y_tensor = torch.FloatTensor(Y_train).to(device)

    dataset = TensorDataset(X_tensor, Y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = model_().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, Y_batch in loader:
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)

        if (epoch + 1) % 2 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')

    return model, losses

def train_surrogate_simple_val(X_train, Y_train, model_=PendulumSurrogate, epochs=1000, lr=0.001, batch_size=128,patience=50,lr_patience=20, lr_factor=0.5, min_lr=1e-6):
    """Train the surrogate model with 80/20 validation split"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    Y_tensor = torch.FloatTensor(Y_train)

    # ── 80/20 Split ──
    dataset    = TensorDataset(X_tensor, Y_tensor)
    total      = len(dataset)
    train_size = int(0.7 * total)
    val_size   = total - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model     = model_().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,  # multiply LR by this on plateau
        patience=lr_patience,  # epochs to wait before reducing
        min_lr=min_lr,  # floor for LR

    )

    train_losses = []
    val_losses   = []
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(epochs):

        # ── Training ──
        model.train()
        epoch_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss   = criterion(Y_pred, Y_batch) # +0.1 * physics_loss(Y_pred, X_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train)

        # ── Validation ──
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                Y_pred = model(X_batch)
                loss   = criterion(Y_pred, Y_batch)  #+0.1 * physics_loss(Y_pred, X_batch)
                epoch_val_loss += loss.item()

        avg_val = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val)

        scheduler.step(avg_val)

        # ── Early Stopping Check ──
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())  # save best weights
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch + 1}/{epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | '
                  f'Best Val: {best_val_loss:.6f} | LR: {current_lr:.2e} | No improve: {epochs_no_improve}/{patience}')

        if epochs_no_improve >= patience:
            print(f'\n Early stopping at epoch {epoch + 1} — val loss stagnant for {patience} epochs.')
            break

        # ── Restore Best Weights ──
    model.load_state_dict(best_model_wts)

    return model, train_losses

