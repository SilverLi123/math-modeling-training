"""Stage audited candidate figures flat for the skill's nonrecursive audit."""
import hashlib
import json
import shutil
from pathlib import Path
from q1_lmm_baseline import write_json

ROOT = Path(__file__).resolve().parents[1]
SETS = ['q1_comparison', 'q2_probability', 'q4_chromosomes', 'paper_coverage']


def main():
    target = ROOT/'paper/figures'
    target.mkdir(parents=True, exist_ok=True)
    previous_path = ROOT/'paper/figure_sources.json'
    previous = {item['destination']: item['sha256'] for item in
                json.loads(previous_path.read_text(encoding='utf-8'))} if previous_path.exists() else {}
    files = []
    for name in SETS:
        source = ROOT/'outputs/figures'/name
        for path in source.glob('*'):
            if path.suffix not in ('.png', '.svg'):
                continue
            dest = target/path.name
            if dest.exists() and dest.read_bytes() != path.read_bytes():
                if hashlib.sha256(dest.read_bytes()).hexdigest() != previous.get(str(dest.relative_to(ROOT))):
                    raise ValueError(f'Untracked change in staged figure: {dest}')
            shutil.copy2(path, dest)
            files.append({'source': str(path.relative_to(ROOT)), 'destination': str(dest.relative_to(ROOT)),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(ROOT/'paper/figure_sources.json', files)
    print(f'Staged {len(files)} PNG/SVG files')


if __name__ == '__main__':
    main()
