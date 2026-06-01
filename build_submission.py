#!/usr/bin/python3
"""Build final self-contained submission.py with embedded model weights."""

import torch, base64, gzip, io

MODEL = "alphazero-board-games/connect4/data/model.1771199333074.pt"

# ---- Step 1: encode model weights ----
ckpt = torch.load(MODEL, weights_only=True, map_location="cpu")
half = {k: v.half() for k, v in ckpt["model"].items()}
buf = io.BytesIO()
torch.save(half, buf)
compressed = gzip.compress(buf.getvalue(), compresslevel=9)
encoded = base64.b64encode(compressed).decode()

print(f"Model blob: {len(encoded):,} chars")

# ---- Step 2: read current submission.py ----
with open("submission.py") as f:
    lines = f.readlines()

# ---- Step 3: rebuild with embedded weights ----
out = []
in_paths_section = False
in_resolve_fn = False
skip_until_blank = False
resolve_indent = 0

for i, line in enumerate(lines):
    # Detect start of paths section — replace everything from here
    # through _resolve_checkpoint() with the embedded blob.
    # BUT stop BEFORE the MCTS/Network config variables.
    if '# Paths' in line and '_CHECKPOINT_PATH' in line:
        in_paths_section = True
        # Replace with embedded blob
        out.append("# ============================================================================\n")
        out.append("# Embedded model weights  (float16 + gzip + base64 — no external file needed)\n")
        out.append("# ============================================================================\n")
        out.append("_MODEL_BLOB = (\n")
        for j in range(0, len(encoded), 120):
            out.append(f"    '{encoded[j:j+120]}'\n")
        out.append(")\n")
        out.append("\n")
        out.append("# MCTS\n")
        out.append("_SIMULATION_NUM = 200\n")
        out.append("_C_PUCT = 1.5\n")
        out.append("_DIRICHLET_ALPHA = 0.3\n")
        out.append("_DIRICHLET_EPSILON = 0.25\n")
        out.append("\n")
        out.append("# Network (must match checkpoint)\n")
        out.append("_CONV_FILTERS = 64\n")
        out.append("_CONV_KERNEL = (3, 3)\n")
        out.append("_RESIDUAL_BLOCK_NUM = 4\n")
        out.append("\n")
        out.append("def _load_embedded_weights(nnet):\n")
        out.append('    """Decode embedded weights and load into the network."""\n')
        out.append("    raw = gzip.decompress(base64.b64decode(''.join(_MODEL_BLOB)))\n")
        out.append("    half_state = torch.load(io.BytesIO(raw), weights_only=True, map_location='cpu')\n")
        out.append("    state = {k: v.float() for k, v in half_state.items()}\n")
        out.append("    nnet.model.load_state_dict(state)\n")
        out.append("\n")
        continue

    if in_paths_section:
        # Skip everything until after _resolve_checkpoint function ends
        if line.startswith('def _resolve_checkpoint'):
            in_resolve_fn = True
            continue
        if in_resolve_fn:
            # Skip function body
            if line.strip() == '' or (line.strip() and not line[0].isspace()):
                if line.strip() == '' or line.startswith('#'):
                    in_resolve_fn = False
                    in_paths_section = False
                    out.append(line)  # keep the blank line after
                continue
            continue
        continue

    # Fix: remove optimizer from _NNet.__init__ (and continuation lines)
    if 'self.optimizer = optim.Adam(' in line:
        skip_until_blank = True
        continue
    if skip_until_blank and line.strip() == ')':
        skip_until_blank = False
        continue
    if skip_until_blank:
        continue

    # Fix: simplify load_checkpoint (no optimizer)
    if 'self.optimizer.load_state_dict(ckpt["optimizer"])' in line:
        continue  # skip optimizer loading

    # Fix: replace file-based checkpoint loading with embedded
    if 'nnet.load_checkpoint(_resolve_checkpoint())' in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append(f'{indent}_load_embedded_weights(nnet)\n')
        continue

    # Fix: smoke-test checkpoint path reference
    if 'print(f"Checkpoint : {_resolve_checkpoint()}")' in line:
        out.append(line.replace(
            'print(f"Checkpoint : {_resolve_checkpoint()}")',
            'print("Checkpoint : embedded (no external file)")'
        ))
        continue

    out.append(line)

# ---- Step 4: add missing imports (io, gzip, base64) ----
result = ''.join(out)
result = result.replace(
    'import glob as _glob\nimport os as _os\n',
    'import io\nimport gzip\nimport os as _os\nimport base64\n',
)

# ---- Write ----
with open("submission.py", "w") as f:
    f.write(result)

print(f"Done! submission.py: {len(result):,} chars, {result.count(chr(10))} lines")
