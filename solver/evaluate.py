"""Evaluate the solver on the train split: python evaluate.py [biz_id ...]"""
import collections
import datetime as dt
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

import periods
from loader import load_business
from qa import answer
from scoring import score

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'train')


def main(argv):
    only = set(a for a in argv if not a.startswith('-'))
    show = '-v' in argv
    periods.load_calendar(os.path.join(ROOT, 'calendar.csv'))
    qs = [json.loads(l) for l in open(os.path.join(ROOT, 'questions.jsonl'), encoding='utf-8')]
    ans = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(ROOT, 'answers.jsonl'), encoding='utf-8')}
    by = collections.defaultdict(list)
    for q in qs:
        if not only or q['business_id'] in only:
            by[q['business_id']].append(q)
    tot = collections.Counter()
    per_h = collections.defaultdict(collections.Counter)
    per_b = {}
    wrong = []
    for bid in sorted(by):
        t0 = time.time()
        try:
            biz = load_business(os.path.join(ROOT, 'data', bid))
        except Exception as e:
            print(bid, 'LOAD FAIL', e)
            traceback.print_exc()
            for q in by[bid]:
                tot['abstain'] += 1
            continue
        c = collections.Counter()
        for q in by[bid]:
            today = dt.date.fromisoformat(q['as_of_date'])
            try:
                pred = answer(biz, q['question'], today)
            except Exception as e:
                pred = {'type': 'abstain', '_h': 'ERROR', '_why': repr(e)}
                if show:
                    traceback.print_exc()
            s = score(pred, ans[q['id']])
            k = {1: 'correct', 0: 'abstain', -1: 'wrong'}[s]
            c[k] += 1
            tot[k] += 1
            per_h[pred.get('_h')][k] += 1
            if s <= 0:
                wrong.append((bid, q, pred, ans[q['id']], k))
        per_b[bid] = c
        n = sum(c.values())
        print(f'{bid}: n={n} correct={c["correct"]} wrong={c["wrong"]} abstain={c["abstain"]} '
              f'PA={(c["correct"] - c["wrong"]) / n:.3f}  ({time.time() - t0:.1f}s)', flush=True)
    n = sum(tot.values())
    print(f'TOTAL n={n} correct={tot["correct"]} wrong={tot["wrong"]} abstain={tot["abstain"]} '
          f'PA={(tot["correct"] - tot["wrong"]) / max(n, 1):.4f}')
    for h, c in sorted(per_h.items(), key=lambda x: -sum(x[1].values())):
        print(f'   {h:24s} correct={c["correct"]:4d} wrong={c["wrong"]:4d} abstain={c["abstain"]:4d}')
    with open(os.path.join(os.path.dirname(__file__), '..', '.cache', 'wrong.jsonl'), 'w', encoding='utf-8') as f:
        for bid, q, pred, ref, k in wrong:
            f.write(json.dumps({'biz': bid, 'k': k, 'q': q['question'], 'asof': q['as_of_date'],
                                'pred': {a: b for a, b in pred.items() if a != 'reason'},
                                'ref': {'type': ref['type'], 'answer': ref.get('answer')}},
                               ensure_ascii=False, default=str) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
