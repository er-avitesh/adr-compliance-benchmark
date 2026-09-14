"""Build read-only, auditable manuscript evidence from the frozen full-text run."""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import adr_scoring as rubric
from scripts.rescore_saved_results import run_path
RUN = ROOT / 'results/runs/v31_fulltext_gpt_medium'
OUT = ROOT / 'results/manuscript_v31/work'
LABELS = ['Not_Compliant', 'Partially_Compliant', 'Mostly_Compliant', 'Fully_Compliant']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score(row):
    result = rubric.evaluate_criteria(row)
    return {'sc': result['sc_score'], 'dq': result['dq_score'],
            'overall': LABELS.index(result['overall']),
            'sc_class': LABELS.index(result['sc_class']),
            'dq_class': LABELS.index(result['dq_class'])}


def alpha(pairs, k, ordinal=True):
    counts = np.bincount(pairs.ravel(), minlength=k).astype(float)
    if ordinal:
        mids = np.cumsum(counts) - counts / 2
        distance = (mids[:, None] - mids[None, :]) ** 2
    else:
        distance = 1 - np.eye(k)
    n = counts.sum()
    expected = np.sum(counts[:, None] * counts[None, :] * distance) / (n * (n - 1))
    return float(1 - np.mean(distance[pairs[:, 0], pairs[:, 1]]) / expected) if expected else None


def reliability(pairs, k, ordinal=True):
    pairs = np.asarray(pairs, int)
    rng = np.random.default_rng(20260909)
    vals = [alpha(pairs[rng.integers(0, len(pairs), len(pairs))], k, ordinal) for _ in range(10000)]
    valid = [x for x in vals if x is not None]
    return {'n': len(pairs), 'matches': int(np.sum(pairs[:, 0] == pairs[:, 1])),
            'agreement': float(np.mean(pairs[:, 0] == pairs[:, 1])), 'alpha': alpha(pairs, k, ordinal),
            'ci95': np.quantile(valid, [.025, .975]).tolist() if valid else None,
            'undefined_bootstrap_samples': 10000 - len(valid)}


def metrics(y, preds):
    out = []
    for pred in preds:
        cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
        den = cm.sum(0) + cm.sum(1)
        f = np.divide(2 * cm.diagonal(), den, out=np.zeros(4), where=den != 0)
        po = np.trace(cm) / cm.sum()
        pe = np.dot(cm.sum(0), cm.sum(1)) / cm.sum() ** 2
        out.append([f.mean(), (po - pe) / (1 - pe), po])
    return np.mean(out, axis=0).tolist()


def main(run_id='v31_fulltext_gpt_medium', output_dir=None):
    global RUN, OUT
    RUN = run_path(run_id)
    manifest = rubric.require_exact_run(RUN)
    OUT = Path(output_dir) if output_dir else RUN / 'analysis/manuscript_evidence'
    OUT.mkdir(parents=True, exist_ok=True)
    batch_run = run_path(manifest['source_run_id']) if manifest.get('analysis_only') else RUN
    source_hashes = read(RUN / 'analysis/exact_rescoring_report.json')['source_sha256'] if manifest.get('analysis_only') else {}
    truth = read(ROOT / 'results/human_ground_truth.json')
    second = read(ROOT / 'results/human_ground_truth_interrator.json')
    ev = read(ROOT / 'results/eval_set.json')['adrs']
    ids = [x['id'] for x in ev]
    y = np.array([LABELS.index(truth[x]['overall']) for x in ids])
    report = {'seed': 20260909, 'bootstrap_resamples': 10000,
              'scoring_version': rubric.SCORING_VERSION, 'analysis_run_id': run_id,
              'inputs': {}, 'reliability': {}, 'configurations': {}, 'costs': {}, 'errors': []}
    for p in [ROOT / 'results/eval_set.json', ROOT / 'results/human_ground_truth.json', ROOT / 'results/human_ground_truth_interrator.json']:
        report['inputs'][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    common = sorted(second)
    assert len(common) == 41 and set(common) <= set(ids)
    assert alpha(np.array([[0, 0], [1, 1]]), 2) == 1
    assert alpha(np.array([[0, 1], [1, 0]]), 2) == -.5
    for key in ['overall', 'sc_class', 'dq_class'] + [f'C{i}' for i in range(1, 8)] + [f'Q{i}' for i in range(1, 8)]:
        if key == 'overall':
            pairs = [[LABELS.index(truth[x][key]), LABELS.index(second[x][key])] for x in common]
        elif key.endswith('_class'):
            pairs = [[score(truth[x])[key], score(second[x])[key]] for x in common]
        else:
            pairs = [[int(truth[x][key]), int(second[x][key])] for x in common]
        report['reliability'][key] = reliability(pairs, 2 if key.startswith('C') else 4, not key.startswith('C'))
    report['reliability']['formula_overall'] = reliability([[score(truth[x])['overall'], score(second[x])['overall']] for x in common], 4)
    report['second_review_formula_discrepancies'] = [x for x in common if LABELS[score(second[x])['overall']] != second[x]['overall']]
    report['second_review_dimension_discrepancies'] = {k: [x for x in common if LABELS[score(second[x])[k]] != second[x].get(k)] for k in ['sc_class', 'dq_class']}
    report['reviewer_counts'] = {'first': dict(Counter(truth[x].get('reviewer_id') for x in ids)), 'second': dict(Counter(second[x].get('reviewer_id') for x in common))}
    report['human_distributions'] = {k: dict(Counter(LABELS[score(truth[x])[k]] for x in ids)) for k in ['overall', 'sc_class', 'dq_class']}
    report['human_criteria_distributions'] = {k: dict(Counter(str(truth[x][k]) for x in ids)) for k in [f'C{i}' for i in range(1, 8)] + [f'Q{i}' for i in range(1, 8)]}
    docs = {x: read(ROOT / 'results/adrs' / (x + '.json')) for x in ids}
    decoded_inputs = {x: json.loads((ROOT / 'results/adrs' / (x + '.json')).read_text(encoding='cp1252'))['text'] for x in ids}
    report['input_encoding_anomalies'] = {
        x: {'utf8_characters': len(docs[x]['text']), 'run_input_characters': len(decoded_inputs[x]),
            'utf8_text_sha256': hashlib.sha256(docs[x]['text'].encode('utf8')).hexdigest(),
            'run_input_sha256': hashlib.sha256(decoded_inputs[x].encode('utf8')).hexdigest()}
        for x in ids if docs[x]['text'] != decoded_inputs[x]}
    excluded_repos = {'alphagov/govuk-aws', 'adr/madr', 'argoproj/argo-cd'}
    masks = {'all': np.arange(len(ids)), 'unseen_exemplar_repository': np.array([i for i, x in enumerate(ev) if x['source_repo'] not in excluded_repos]), 'same_exemplar_repository': np.array([i for i, x in enumerate(ev) if x['source_repo'] in excluded_repos])}
    masks['encoding_unaffected'] = np.array([i for i, x in enumerate(ids) if x not in report['input_encoding_anomalies']])
    raw_map = {}
    gemini_usage = {}
    for path in (batch_run / 'batch_jobs/outputs').glob('gemini*.jsonl'):
        relative = str(path.relative_to(ROOT))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hashes and source_hashes.get(relative) != sha:
            raise ValueError(f'Original batch evidence changed: {relative}')
        report['inputs'][relative] = sha
        for line in path.read_text(encoding='utf8').splitlines():
            item = json.loads(line)
            gemini_usage[item['key']] = item['response']['usageMetadata']
    total_rows = errors = parse_failures = truncated = 0
    for path in sorted((RUN / 'raw_results').glob('*.json')):
        if path.name == 'all_results.json':
            continue
        raw = read(path)
        if not isinstance(raw, list) or len(raw) != 3:
            continue
        rows = [r for rep in raw for r in rep]
        report['inputs'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in rows:
            fields = rubric.prediction_fields(row['predicted_sc_checks'], row['predicted_dq_scores'])
            if any(row.get(key) != value for key, value in fields.items()):
                raise ValueError(f"Stale scoring for {row['adr_id']}")
        name = rows[0]['model_key'] + '/' + rows[0]['strategy']
        raw_map[name] = [[{r['adr_id']: r for r in rep}[x] for x in ids] for rep in raw]
        preds = np.array([[LABELS.index(r['predicted']) for r in rep] for rep in raw_map[name]])
        assert all(len(rep) == len(ids) for rep in raw)
        assert all(r['adr_input_sha256'] == hashlib.sha256(decoded_inputs[r['adr_id']].encode('utf8')).hexdigest() for r in rows)
        total_rows += len(rows)
        errors += sum(bool(r.get('error')) for r in rows)
        parse_failures += sum(not r.get('criterion_parse_success') for r in rows)
        truncated += sum(r['input_was_truncated'] or r['adr_input_chars'] != r['adr_source_chars'] for r in rows)
        report['configurations'][name] = {'subgroups': {key: {'n': len(mask), 'class_counts': dict(Counter(LABELS[v] for v in y[mask])), 'macro_f1_kappa_accuracy': metrics(y[mask], preds[:, mask])} for key, mask in masks.items()}}
        inp = output = thinking = batch = sync = retries = 0
        cost = 0.
        for r in rows:
            i, o = r['input_tokens'], r['output_tokens']
            if r['model_key'] == 'gemini-2.5-pro':
                u = gemini_usage[r['batch_metadata']['custom_id']]
                assert i == u['promptTokenCount'] and o == u['candidatesTokenCount']
                thinking += u.get('thoughtsTokenCount', 0)
                o += u.get('thoughtsTokenCount', 0)
                assert i + o == u['totalTokenCount']
            inp += i
            output += o
            batch += r.get('billing_mode') == 'batch'
            sync += r.get('billing_mode') != 'batch'
            retries += r.get('output_retry_count', 0)
            md = r['model_metadata']
            cost += (i * md['cost_assumption_usd_per_million_input_tokens'] + o * md['cost_assumption_usd_per_million_output_tokens']) / 1e6 * r.get('cost_multiplier', 1)
        report['costs'][name] = {'input_tokens': inp, 'billable_output_tokens': output, 'gemini_thinking_tokens': thinking, 'batch': batch, 'sync': sync, 'output_retries': retries, 'estimated_usd': cost, 'per_evaluation_usd': cost/486}
    report['run_validation'] = {'configurations': len(raw_map), 'rows': total_rows, 'api_errors': errors, 'criterion_parse_failures': parse_failures, 'truncated_rows': truncated}
    example_raw = next(iter(raw_map.values()))[0]
    report['input_lengths'] = {'max': max(r['adr_source_chars'] for r in example_raw), 'over_3500': sum(r['adr_source_chars'] > 3500 for r in example_raw)}
    # Both configuration assignments and all repetitions stay together per repository.
    repos = sorted({x['source_repo'] for x in ev})
    groups = [np.array([i for i, x in enumerate(ev) if x['source_repo'] == repo]) for repo in repos]
    a, b = 'gpt-5.5/few_shot', 'gemini-2.5-pro/few_shot'
    pa = np.array([[LABELS.index(r['predicted']) for r in rep] for rep in raw_map[a]])
    pb = np.array([[LABELS.index(r['predicted']) for r in rep] for rep in raw_map[b]])
    rng = np.random.default_rng(20260909)
    differences = []
    for _ in range(10000):
        ix = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        differences.append(np.array(metrics(y[ix], pa[:, ix])) - np.array(metrics(y[ix], pb[:, ix])))
    report['repository_block_bootstrap_top_pair'] = {'a': a, 'b': b, 'n_repositories': len(repos), 'ci95_macro_f1_kappa_accuracy': np.quantile(differences, [.025, .975], axis=0).T.tolist(), 'caution': 'Exploratory repository resampling; 14 repositories, not a population-random sample.'}
    for i, x in enumerate(ids):
        if pa[0, i] != y[i]:
            r = raw_map[a][0][i]
            report['errors'].append({'adr_id': x, 'human_overall': LABELS[y[i]], 'predicted_overall': r['predicted'], 'human_sc_dq': [score(truth[x])['sc'], score(truth[x])['dq']], 'model_sc_dq': [r['predicted_sc_score'], r['predicted_dq_score']], 'criterion_differences': {k: [truth[x][k], r['predicted_sc_checks' if k[0] == 'C' else 'predicted_dq_scores'][k]] for k in [f'C{i}' for i in range(1,8)]+[f'Q{i}' for i in range(1,8)] if truth[x][k] != r['predicted_sc_checks' if k[0] == 'C' else 'predicted_dq_scores'][k]}})
    (OUT / 'evidence.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({k: report[k] for k in ['run_validation', 'input_lengths', 'second_review_formula_discrepancies', 'second_review_dimension_discrepancies', 'repository_block_bootstrap_top_pair']}, indent=2))
    print('Human overall:', report['reliability']['overall'])
    print('Cost total:', sum(x['estimated_usd'] for x in report['costs'].values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    main(args.run_id, args.output_dir)
