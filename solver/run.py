"""Answer every question of a split and write submission.jsonl.

usage: python solver/run.py --data data/public_test --out output/submission.jsonl [--logs output/logs]
"""
import argparse
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

# Handlers whose train precision is too low to risk a -1: their answers become abstain.
POLICY_FILE = os.path.join(os.path.dirname(__file__), 'policy.json')


def load_policy():
    if os.path.exists(POLICY_FILE):
        with open(POLICY_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {'abstain_handlers': []}


def to_submission(qid, pred):
    t = pred.get('type')
    if t == 'unanswerable':
        return {'id': qid, 'type': 'unanswerable', 'reason': str(pred.get('reason', ''))}
    if t in (None, 'abstain'):
        return {'id': qid, 'type': 'abstain'}
    return {'id': qid, 'type': t, 'answer': pred.get('answer')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--logs', default=None)
    ap.add_argument('--no-policy', action='store_true')
    args = ap.parse_args()

    periods.load_calendar(os.path.join(args.data, 'calendar.csv'))
    policy = {'abstain_handlers': []} if args.no_policy else load_policy()
    skip = set(policy.get('abstain_handlers', []))
    qs = [json.loads(l) for l in open(os.path.join(args.data, 'questions.jsonl'), encoding='utf-8')]
    by = collections.defaultdict(list)
    for q in qs:
        by[q['business_id']].append(q)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.logs:
        os.makedirs(args.logs, exist_ok=True)
    results = {}
    logs = []
    stats = collections.Counter()
    for bid in sorted(by):
        t0 = time.time()
        try:
            biz = load_business(os.path.join(args.data, 'data', bid))
            load_err = None
        except Exception as e:
            biz, load_err = None, repr(e)
            traceback.print_exc()
        load_t = time.time() - t0
        for q in by[bid]:
            t1 = time.time()
            if biz is None:
                pred = {'type': 'abstain', '_h': 'LOAD_FAIL', '_why': load_err}
            else:
                try:
                    pred = answer(biz, q['question'], dt.date.fromisoformat(q['as_of_date']))
                except Exception as e:
                    pred = {'type': 'abstain', '_h': 'ERROR', '_why': repr(e)}
            if pred.get('_h') in skip and pred.get('type') not in ('abstain',):
                pred = dict(pred, type='abstain', _policy='abstain')
            sub = to_submission(q['id'], pred)
            results[q['id']] = sub
            stats[sub['type']] += 1
            logs.append({'id': q['id'], 'business_id': bid, 'question': q['question'], 'as_of_date': q['as_of_date'],
                         'final': sub, 'handler': pred.get('_h'), 'note': pred.get('_why') or pred.get('reason'),
                         'elapsed_s': round(time.time() - t1 + (load_t if q is by[bid][0] else 0), 4)})
        print(f'{bid}: {len(by[bid])} questions, load {load_t:.1f}s', flush=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for q in qs:
            f.write(json.dumps(results[q['id']], ensure_ascii=False) + '\n')
    if args.logs:
        with open(os.path.join(args.logs, 'answers_log.jsonl'), 'w', encoding='utf-8') as f:
            for r in logs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
    print('types:', dict(stats))


if __name__ == '__main__':
    main()
