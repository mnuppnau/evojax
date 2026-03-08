import numpy as np
import sys

data = np.load(sys.argv[1])
n_classes = 10

prototypes = []
for i in range(n_classes):
    imgs = data[i:60:10, :, :, 0]
    prototypes.append(np.mean(imgs, axis=0))
prototypes = np.array(prototypes)

dists = np.zeros((n_classes, n_classes))
for i in range(n_classes):
    for j in range(n_classes):
        dists[i,j] = np.sqrt(np.sum((prototypes[i] - prototypes[j])**2))

# Find the closest pairs
flat_dists = []
for i in range(n_classes):
    for j in range(i+1, n_classes):
        flat_dists.append((dists[i,j], i, j))

flat_dists.sort()
print("Top 5 closest code pairs (L2 Pixel Distance):")
for d, i, j in flat_dists[:5]:
    print(f"Code {i} and Code {j}: {d:.2f}")

