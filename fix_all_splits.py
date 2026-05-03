import os
import glob

abstasks_path = 'third_party/FinMTEB/finance_mteb/abstasks/'
files = glob.glob(abstasks_path + 'AbsTask*.py')

def make_fallback(d):
    return f'next((s for s in [split, "test", "dev", "validation", "train"] if s in {d}), split)'

# Add more data attributes if needed
attributes = [
    'self.corpus',
    'self.queries',
    'self.relevant_docs',
    'self.dataset',
    'self.data',
]

for filepath in files:
    with open(filepath, 'r') as f:
        content = f.read()

    original = content
    total = 0

    for attr in attributes:
        old = f'{attr}[split]'
        new = f'{attr}[{make_fallback(attr)}]'
        count = content.count(old)
        if count > 0:
            content = content.replace(old, new)
            total += count
            print(f"  [{os.path.basename(filepath)}] Replaced {count}x: {old}")

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✓ Patched {os.path.basename(filepath)} ({total} replacements)\n")
    else:
        print(f"- Skipped {os.path.basename(filepath)} (no changes needed)\n")

print("All done!")
