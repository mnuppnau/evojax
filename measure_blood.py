import numpy as np
import sys

data = np.load(sys.argv[1])
n_classes = 8
prototypes = []
for i in range(n_classes):
    imgs = data[i:data.shape[0]:n_classes, :, :, 0]
    prototypes.append(np.mean(imgs, axis=0))

prototypes = np.array(prototypes)

dists = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        dists[i,j] = np.sqrt(np.sum((prototypes[i] - prototypes[j])**2))

flat_dists = []
for i in range(n_classes):
    for j in range(i+1, n_classes):
        flat_dists.append((dists[i,j], i, j))

flat_dists.sort()
print("Top 5 closest code pairs (L2 Pixel Distance):")
for d, i, j in flat_dists[:5]:
    print(f"Code {i} and Code {j}: {d:.2f}")

print("Top 5 furthest code pairs:")
for d, i, j in flat_dists[-5:]:
    print(f"Code {i} and Code {j}: {d:.2f}")

intra_vars = []
for i in range(n_classes):
    imgs = data[i:data.shape[0]:n_classes, :, :, 0]
    intra_vars.append(np.mean(np.var(imgs, axis=0)))

print("--- Intra-Code Pixel Variance ---")
for i, var in enumerate(intra_vars):
    print(f"Code {i:2d}: {var:.4f}")
