"""Table 2 of paper v8: every reading on the three metrics, with the verdict against received attention.
Reads results/table1_long.csv, results/correct_tests.csv and results/variants.csv; values are only rounded.
Usage: python3 -I make_main_table.py <repo results dir> <output tex>"""
import csv, sys
res, out = sys.argv[1], sys.argv[2]
val = {(r['model'], r['metric'], r['method']): float(r['mean']) for r in csv.DictReader(open(res + '/table1_long.csv'))}
verdict = {}
for r in csv.DictReader(open(res + '/correct_tests.csv')):
    if r['b'] == 'R0' and r['test'] in ('correction - R0', 'stand-in - R0'):
        verdict[(r['model'], r['metric'], r['a'])] = r['verdict']
var_val, var_verdict = {}, {}
for r in csv.DictReader(open(res + '/variants.csv')):
    if r['test'] == 'value': var_val[(r['variant'], r['metric'])] = float(r['mean'])
    else: var_verdict[(r['variant'], r['metric'])] = r['verdict']
MODELS = ['deit', 'deit_b', 'dino']; METRICS = ['enrichment', 'clickme', 'srg']
ARROW = {'higher': r'\up', 'lower': r'\dn', 'comparable': ''}
def num(x):
    s = f'{abs(x):.3f}'
    return ('$-$' if round(x, 3) < 0 else '') + s
def row(label, method, test=True):
    cells = []
    for m in MODELS:
        for k in METRICS:
            a = ARROW[verdict[(m, k, method)]] if test else ''
            cells.append(num(val[(m, k, method)]) + a)
    return label + ' & ' + ' & '.join(cells) + r' \\'
def vrow(label, variant):
    cells = []
    for k in METRICS:
        cells.append(num(var_val[(variant, k)]) + ARROW[var_verdict[(variant, k)]] if (variant, k) in var_val else '--')
    return label + ' & ' + ' & '.join(cells + ['--'] * 6) + r' \\'
L = [r'\begin{tabular}{@{}l@{\hspace{4pt}}ccc@{\hspace{7pt}}ccc@{\hspace{7pt}}ccc@{}}', r'\toprule',
     r' & \multicolumn{3}{c}{DeiT-S} & \multicolumn{3}{c}{DeiT-B} & \multicolumn{3}{c}{DINO ViT-S}\\',
     r'\cmidrule(r{6pt}){2-4}\cmidrule(r{6pt}){5-7}\cmidrule{8-10}',
     'Reading' + ' & Enr. & Click. & Del.' * 3 + r'\\', r'\midrule',
     row('Received attention', 'R0', test=False),
     row('Contribution weighting', 'C3'), row(r'\quad zero high-norm tokens', 'stand_in_no_high'),
     row('Mean-map subtraction', 'C1'), row(r'\quad centre-Gaussian subtraction', 'stand_in_centre'),
     row('Probe relocation', 'C2'), row(r'\quad matched blur', 'stand_in_blur'),
     row('All three corrections', 'R_all'), r'\midrule',
     vrow('+ test-time register', 'register'), vrow('Top-$k$ training (MiTA)', 'mita'),
     row('GAE (reference)', 'gae', test=False), r'\bottomrule', r'\end{tabular}']
open(out, 'w').write('% Table 1: made by make/make_main_table.py from results/table1_long.csv, correct_tests.csv and variants.csv.\n'
                     + '\n'.join(L) + '\n')
print('\n'.join(L[6:14] + L[15:18]))
