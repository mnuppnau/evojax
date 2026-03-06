import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Pattern to find the global average pool and replace it with 7x7 spatial flattened conv
    old_q_trunk = r"""        # ----- Q trunk .*?-----
        q = SN\(nn\.Conv\(
            self\.features \* 4,
            kernel_size=\(3, 3\),
            strides=\(1, 1\),
            padding="SAME",
            kernel_init=normal_init\(0\.02\),
        \)\)\(h, update_stats=train\)  # 7x7 -> 7x7
        q = nn\.leaky_relu\(q, 0\.2\)

        q_feat_avg = jnp\.mean\(q, axis=\(1, 2\)\)      # \(B, features\*4\)
        q_flat = q_feat_avg                          # \(B, features\*4\)"""

    new_q_trunk = """        # ----- Q trunk -----
        # 7x7 -> 1x1 (Ensures Q-head also cares about spatial arrangement)
        q = SN(nn.Conv(
            self.features * 4,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding="VALID",
            kernel_init=normal_init(0.02),
        ))(h, update_stats=train)
        q = nn.leaky_relu(q, 0.2)
        
        q_flat = q.reshape((q.shape[0], -1))  # (B, features*4)
        q_feat_avg = q_flat"""

    new_content = re.sub(old_q_trunk, new_q_trunk, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Fixed Q trunk in {filepath}")
    else:
        print(f"Pattern not found in {filepath}")

fix_file('evojax/policy/convnet.py')
fix_file('evojax/trainer.py')
