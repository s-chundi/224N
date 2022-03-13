import csv
import util
from ujson import load as json_load

word = 'how'
with open('data/dev_eval.json', 'r') as fh:
        gold_dict = json_load(fh)

preds = {}
uuids = []
reversed_gold = {v['uuid']: k for k, v in gold_dict.items()}

i = 0
# Uuids of when questions
for cq in gold_dict.values():
    if word in cq['question'].lower():
        uuids.append(cq['uuid'])
        i += 1

print(f'There are {i} questions that have \'{word}\' in them')

f = open('data/submission.csv')
csv_f = csv.reader(f)
for id_ans in csv_f:
    if id_ans[0] in uuids:
        ans = id_ans[1]
        qnum = reversed_gold[id_ans[0]]
        preds[qnum] = ans

print(util.eval_dicts(gold_dict, preds, True))