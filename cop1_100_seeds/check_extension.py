#!/usr/bin/env python3
"""Report completion of the 12 batches without using GPU dependencies."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path, help='.../af3_batch/double100-ARRAY_JOB_ID')
args = parser.parse_args()
total = 0
bad = False
for construct in ('dog9A', 'dog11A', 'dog13A'):
    count = 0
    for first in (21, 41, 61, 81):
        path = args.root / construct / 'seeds{}-{}'.format(first, first + 19) / 'extension_verified.json'
        try:
            report = json.loads(path.read_text())
            if not (report.get('verified') and report['construct'] == construct
                    and report['first_seed'] == first and report['last_seed'] == first + 19
                    and report['model_count'] == 20):
                raise ValueError('Unexpected verification contents')
            count += 20
        except (OSError, ValueError, KeyError) as error:
            print('{} seeds {}-{}: not verified ({})'.format(construct, first, first + 19, error))
            bad = True
    total += count
    print('{}: {}/80 NEW models verified'.format(construct, count))
print('TOTAL: {}/240 NEW models verified'.format(total))
if not bad:
    print('All new batches complete. Combine with the original 20 per construct to obtain 100 each.')
raise SystemExit(1 if bad else 0)
