from calendar import c
import csv
import util
from ujson import load as json_load
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

word = 'how'
with open('data/dev_eval.json', 'r') as fh:
        gold_dict = json_load(fh)

inds = []
results = []
reversed_gold = {v['uuid']: k for k, v in gold_dict.items()}

i = 0


# print(f'There are {i} questions that have \'{word}\' in them')
'''
f = open('data/submission.csv')
csv_f = csv.reader(f)
for id_ans in csv_f:
    if id_ans[0] in uuids:
        ans = id_ans[1]
        qnum = reversed_gold[id_ans[0]]
        preds[qnum] = ans
'''
# Uuids of when questions
for j in tqdm(range(150)):
    preds = {}
    uuids = []
    for cq in gold_dict.values():
        if len(cq['question'].split()) == j:
            uuids.append(cq['uuid'])
            i += 1
            if j < 5:
                print(cq['question'])
    f = open('data/Bidaf.csv')
    csv_f = csv.reader(f)
    for id_ans in csv_f:
        if id_ans[0] in uuids:
            ans = id_ans[1]
            qnum = reversed_gold[id_ans[0]]
            preds[qnum] = ans
    if len(preds) == 0:
        pass
    else:
        eval = util.eval_dicts(gold_dict, preds, True)
        inds.append(j)
        results.append(eval['F1'])

print(f'F1 score for all lengths of questions: {results}')
title = 'F1 by question length (BiDAF baseline)'
plt.scatter(inds, results)
plt.title(title)
plt.xlabel('Question Length')
plt.ylabel('F1 score')
plt.savefig(f'save/graphs/{title}.png')
