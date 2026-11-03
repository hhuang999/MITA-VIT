"""Numbers for the slide charts, read from the result files (same sources as scripts/paper_assets.py).

Usage: python slides/slide_data.py  ->  slides/data.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from paper_assets import MODELS, SUP, v3_values  # noqa: E402


def main():
    values = v3_values()
    methods = ['probe_mass', 'attention_mass', 'cls_attention', 'gae']
    out = {'models': ['DeiT-S', 'DINO ViT-S', 'DeiT-B']}
    for measure in ['srg', 'human', 'head', 'ring']:
        out[measure] = {m: [round(float(values[(measure, model)][m]), 4) for model in MODELS] for m in methods + ['centre']}
    s3 = pd.read_csv(SUP / 's3_layers.csv').query("family == 'probe' and measure == 'high_share'")
    out['high_share'] = {model: s3.query('model == @model').sort_values('layer')['mean'].round(3).tolist()
                         for model in MODELS}
    curves = pd.read_csv(SUP / 's2_layers.csv').query("measure == 'clickme_specific_pixel' and model == 'deit'")
    out['clickme_deit'] = {family: curves.query('family == @family').sort_values('layer')['mean'].round(3).tolist()
                           for family in ['probe', 'attention', 'cls']}
    (ROOT / 'slides' / 'data.json').write_text(json.dumps(out, indent=1) + '\n')
    # The layer figure without its high-norm column (that has its own slide).
    figure = Image.open(ROOT / 'paper' / 'figures' / 'layers.png')
    figure.crop((0, 0, round(figure.width * 0.745), figure.height)).save(ROOT / 'slides' / 'layers_three.png')
    print(json.dumps(out)[:600])


if __name__ == '__main__':
    main()
