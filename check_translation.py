import numpy as np
import sys

data = np.load(sys.argv[1])
print('Centroids of darkness (y, x):')
n = 8
for i in range(n):
    imgs = data[i:data.shape[0]:n, :, :, 0]
    mean_img = np.mean(imgs, axis=0)
    dark = 1.0 - mean_img
    total = np.sum(dark)
    y, x = np.indices((28, 28))
    cy = np.sum(y * dark) / total
    cx = np.sum(x * dark) / total
    print(f"Code {i}: cy={cy:.1f}, cx={cx:.1f}")
