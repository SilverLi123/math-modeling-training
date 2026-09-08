"""Run the installed dual-source search and retain discovery metadata."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUERIES = {'fetal_fraction': 'Gestational age maternal weight fetal cell free DNA',
           'mixed_models': 'Random effects models longitudinal data Laird Ware 1982',
           'nipt_labels': 'Improving calling non invasive prenatal testing trisomy support vector machine Yang',
           'bootstrap': 'Bootstrap methods another look jackknife Efron 1979'}


def main(skill_root):
    out = ROOT/'paper/reference_search'; out.mkdir(parents=True, exist_ok=True)
    script = skill_root/'tools/paper_search/scripts/hybrid_scholar.py'
    for name, query in QUERIES.items():
        result = subprocess.run([sys.executable, str(script), '--query', query, '--limit', '3', '--json'],
                                capture_output=True, encoding='utf-8', check=True)
        data = json.loads(result.stdout)
        (out/f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, data['stats'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--skill-root', type=Path, required=True)
    main(parser.parse_args().skill_root)
