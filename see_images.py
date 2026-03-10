import numpy as np
import sys

filename = sys.argv[1]
n_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
data = np.load(filename)

chars = " .:-=+*#%@"
def to_ascii(img):
    img = np.clip(img, 0, 1)
    res = ""
    # downsample to 14x14 for terminal display
    for r in range(0, 28, 2):
        for c in range(0, 28, 2):
            val = np.mean(img[r:r+2, c:c+2])
            idx = int(val * 9.99)
            res += chars[idx] * 2
        res += "\n"
    return res

for i in range(n_classes):
    print(f"--- Code {i} Prototype ---")
    imgs = data[i:data.shape[0]:n_classes, :, :, 0]
    mean_img = np.mean(imgs, axis=0)
    print(to_ascii(mean_img))

