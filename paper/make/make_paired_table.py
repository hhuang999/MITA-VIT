"""Appendix table of paired differences: each correction against received attention and against its
matched control. Blocks are copied verbatim from paper/tables/appendix/paired_differences.tex
(only the block titles are simplified); no number is recomputed.
Usage: python3 -I make_paired_table.py <repo paper/tables dir> <output tex>"""
import re, sys, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
p = (src / 'appendix' / 'paired_differences.tex').read_text().splitlines()
def block(title_part, title):
    i = next(k for k, l in enumerate(p) if 'multicolumn' in l and title_part in l)
    return [r'\multicolumn{4}{l}{\textit{' + title + r'}} \\'] + p[i + 1:i + 4]
B = [block('Contribution weighting $-$ received attention', 'Contribution weighting $-$ received attention'),
     block('Mean-map subtraction $-$ received attention', 'Mean-map subtraction $-$ received attention'),
     block('Probe relocation $-$ received attention', 'Probe relocation $-$ received attention'),
     block('All three corrections (order A) $-$ received attention', 'All three corrections $-$ received attention'),
     block('Contribution weighting $-$ zero high-norm tokens', 'Contribution weighting $-$ zero high-norm tokens (its control)'),
     block('Mean-map subtraction $-$ centre-Gaussian subtraction', 'Mean-map subtraction $-$ centre-Gaussian subtraction (its control)'),
     block('Probe relocation $-$ matched blur', 'Probe relocation $-$ matched blur (its control)')]
T = ['% Paired differences: blocks copied verbatim from repo/paper/tables/appendix/paired_differences.tex by',
     '% make/make_paired_table.py (titles simplified, numbers unchanged).',
     r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.92}', r'\begin{tabular}{lccc}', r'\toprule',
     r'Model & Same-ring enrichment & ClickMe alignment & Deletion faithfulness \\', r'\midrule']
def reorder(line):   # source column order is deletion, enrichment, ClickMe; the paper lists deletion last
    if 'multicolumn' in line: return line
    body = line.rstrip(); assert body.endswith(r'\\'); body = body[:-2]
    c = [x.strip() for x in body.split('&')]; assert len(c) == 4, line
    return f'{c[0]} & {c[2]} & {c[3]} & {c[1]} \\\\'
for k, bl in enumerate(B):
    if k == 4: T.append(r'\midrule')
    elif k: T.append(r'\addlinespace[2pt]')
    T += [reorder(l) for l in bl]
T += [r'\bottomrule', r'\end{tabular}']
out.write_text('\n'.join(T) + '\n', encoding='utf-8', newline='\n')
num = re.compile(r'\d+\.\d+')
srcnums = set(num.findall('\n'.join(p)))
body = '\n'.join(l for l in T if not l.startswith('%') and 'arraystretch' not in l)
print('rows', body.count('\\\\'), 'numbers not in source:', sorted(set(n for n in num.findall(body) if n not in srcnums)))
