# %%
# 运行环境与随机种子。
from pathlib import Path
import os
# Anaconda 的 NumPy 使用 MKL；先让 MKL 使用顺序计算，避免与 PyTorch 的 OpenMP 冲突。
# PyTorch 自身仍使用下面设置的四个线程。这不放宽重复运行库检查。
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT = PROJECT / 'results' / 'latest'
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(42)
rng = np.random.default_rng(42)
D, N, WIDTH = 100, 10000, 512
print('Python / PyTorch ready. PyTorch:', torch.__version__, flush=True)

# %%
# 训练数据：球内体积均匀采样与平方和标签。
def sample_ball(n, d, generator):
    z = generator.normal(size=(n, d))
    direction = z / np.linalg.norm(z, axis=1, keepdims=True)
    radius = generator.random((n, 1)) ** (1 / d)
    return (direction * radius).astype(np.float32)

X = sample_ball(N, D, rng)
y = np.sum(X**2, axis=1, keepdims=True)
r = np.linalg.norm(X, axis=1)
assert X.shape == (N, D) and y.shape == (N, 1)
assert np.all(r <= 1 + 1e-6)
assert abs(y.mean() - D / (D + 2)) < 0.002
print('Data shape:', X.shape, 'Labels:', y.shape, flush=True)
print('Radius minimum / mean / maximum:', r.min(), r.mean(), r.max(), flush=True)
print('Count below radius 0.9 / 0.95:', (r < .9).sum(), (r < .95).sum(), flush=True)

# %%
# 训练半径和标签分布。
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].hist(r, bins=35, color='steelblue')
axes[0].set(xlabel='Radius', ylabel='Sample count', title='Training radius distribution')
axes[1].hist(y[:, 0], bins=35, color='darkorange')
axes[1].set(xlabel='Target value', ylabel='Sample count', title='Training target distribution')
fig.savefig(OUT / 'training_data.png', dpi=160)
plt.show()
plt.close(fig)

# %%
# 模型结构与初始化。
train_x, train_y = torch.from_numpy(X), torch.from_numpy(y)
model = nn.Sequential(nn.Linear(D, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 1))
loss_fn = nn.MSELoss()
print(model, flush=True)
print('Trainable parameters:', sum(p.numel() for p in model.parameters()), flush=True)
with torch.no_grad():
    initial_mse = loss_fn(model(train_x), train_y).item()
print('Before training MSE:', initial_mse, flush=True)

# %%
# 全批量优化；仅依据训练损失停止。
TARGET = 1e-6
MAX_STEPS = 6000
history = []
start = time.perf_counter()
best_mse = initial_mse
best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, min_lr=1e-5)
for step in range(1, MAX_STEPS + 1):
    optimizer.zero_grad()
    prediction = model(train_x)
    loss = loss_fn(prediction, train_y)
    loss.backward()
    optimizer.step()
    if step % 10 == 0:
        with torch.no_grad():
            mse = loss_fn(model(train_x), train_y).item()
        if not np.isfinite(mse):
            raise RuntimeError('Non-finite training loss')
        scheduler.step(mse)
        history.append({'stage':'Adam', 'step':step, 'mse':mse, 'lr':optimizer.param_groups[0]['lr']})
        if mse < best_mse:
            best_mse = mse
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if step % 100 == 0 or mse < TARGET:
            print(f'Step {step}: training MSE={mse:.8g}; elapsed={time.perf_counter()-start:.1f}s', flush=True)
        if step % 500 == 0:
            torch.save(best_state, OUT / 'training_checkpoint.pt')
        if mse < TARGET:
            break
model.load_state_dict(best_state)
model.eval()
training_seconds = time.perf_counter() - start
with torch.no_grad():
    train_mse = loss_fn(model(train_x), train_y).item()
print('FINAL TRAIN MSE:', train_mse, 'target reached:', train_mse < TARGET, flush=True)
if train_mse >= TARGET:
    print('Threshold NOT reached. Results describe this run only.', flush=True)

# %%
# 独立评估。
def predict(a):
    with torch.no_grad():
        return model(torch.from_numpy(a.astype(np.float32))).numpy().reshape(-1)

# 轴向切片评估：500 个点。
t = np.linspace(-1, 1, 500, dtype=np.float32)
axis_x = np.zeros((500, D), dtype=np.float32)
axis_x[:, 0] = t
axis_pred = predict(axis_x)

# 径向评估：每个半径 1000 个球面点，计算 MAE。
test_rng = np.random.default_rng(1042)
radii = np.linspace(0, 1, 21)
radial_mae = []
for radius in radii:
    z = test_rng.normal(size=(1000, D))
    sphere = radius * z / np.linalg.norm(z, axis=1, keepdims=True)
    radial_mae.append(float(np.mean(np.abs(predict(sphere) - radius**2))))

# 独立同分布测试集。
test_x = sample_ball(10000, D, test_rng)
test_y = np.sum(test_x**2, axis=1)
test_prediction = predict(test_x)
metrics = {
    'initial_mse': initial_mse, 'training_mse': train_mse,
    'threshold_reached': bool(train_mse < TARGET),
    'iid_test_mse': float(np.mean((test_prediction-test_y)**2)),
    'iid_test_mae': float(np.mean(np.abs(test_prediction-test_y))),
    'origin_prediction': float(predict(np.zeros((1,D), dtype=np.float32))[0]),
    'radius_0_5_mae': radial_mae[10], 'radius_1_mae': radial_mae[-1],
    'axis_left_prediction': float(axis_pred[0]),
    'axis_right_prediction': float(axis_pred[-1]),
    'training_seconds': training_seconds,
}
print(json.dumps(metrics, indent=2), flush=True)
fig, ax = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
ax[0].semilogy([h['step'] for h in history], [h['mse'] for h in history])
ax[0].axhline(TARGET, color='tomato', linestyle='--', label='Stopping threshold')
ax[0].set(xlabel='Parameter update', ylabel='Training MSE', title='Training convergence')
ax[0].legend()
ax[1].plot(t, t*t, 'b-', label='Target')
ax[1].plot(t, axis_pred, 'r--', label='Prediction')
ax[1].set(xlabel='First coordinate', ylabel='Function value', title='Axis slice')
ax[1].legend()
ax[2].plot(radii, radial_mae, 'o-', color='tomato')
ax[2].set(xlabel='Radius', ylabel='Mean absolute error', title='Radial evaluation')
for a in ax: a.grid(alpha=.25)
fig.savefig(OUT / 'network_results.png', dpi=170)
plt.show()
plt.close(fig)

# %%
# 归档指标、参数与原始评估数据。
config = {'d': D, 'n': N, 'width': WIDTH, 'seed':42, 'loss':'MSE',
          'training':'Full batch Adam, initial lr=0.001, halve lr after 21 non-improving checkpoints, max 6000 steps',
          'target':TARGET, 'torch':str(torch.__version__), 'numpy':np.__version__,
          'device':'cpu', 'threads':4}
(OUT / 'metrics.json').write_text(json.dumps({'config':config, 'metrics':metrics}, indent=2), encoding='utf-8')
(OUT / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
np.savez(OUT / 'evaluation_data.npz', t=t, axis_prediction=axis_pred, radii=radii, radial_mae=radial_mae)
torch.save(model.state_dict(), OUT / 'model_state.pt')
print('Saved results to', OUT, flush=True)
