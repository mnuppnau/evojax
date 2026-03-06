import numpy as np

data = np.load('iteration-1000.npy')
print(f"Loaded data shape: {data.shape}")

n_classes = 10
viz_repeats = data.shape[0] // n_classes

print(f"Assuming {n_classes} classes and {viz_repeats} repeats.")

for c in range(n_classes):
    # The codes are interleaved: 0, 1, 2...9, 0, 1... so code c is at indices c, c+10, c+20...
    indices = np.arange(c, data.shape[0], n_classes)
    imgs = data[indices]
    mean_val = np.mean(imgs)
    var_val = np.var(imgs)
    min_val = np.min(imgs)
    max_val = np.max(imgs)
    # Check what % of pixels are at the extreme
    sat_pos = np.mean(imgs > 0.95)
    sat_neg = np.mean(imgs < -0.95)
    print(f"Code {c:2d}: mean={mean_val:6.3f} var={var_val:6.3f} min={min_val:6.3f} max={max_val:6.3f}  sat+(>0.95)={sat_pos:5.1%} sat-(<-0.95)={sat_neg:5.1%}")

