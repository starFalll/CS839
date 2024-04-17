import numpy as np
import os
import re
#from utils import canonical_header

def canonical_header(h, max_header_len=30):
    # convert any header to its canonincal form
    # e.g. fileSize
    h = str(h)
    if len(h)> max_header_len:
        return '-'
    h = re.sub(r'\([^)]*\)', '', h) # trim content in parentheses
    h = re.sub(r"([A-Z][a-z])", r" \1", h) #insert a space before any Cpital starts
    words = list(filter(lambda x: len(x)>0, map(lambda x: x.lower(), re.split('\W', h))))
    if len(words)<=0:
        return '-'
    new_phrase = ''.join([words[0]] + [x.capitalize() for x in words[1:]])
    return new_phrase

if __name__ == "__main__":
    predlist = []
    truelist = []
    for i in range(5):
        p_arr = np.load(f'npy_single/preds/K{i}_pred.npy')
        for guess in p_arr:
            predlist.append(canonical_header(guess))
        truelist.append(np.load(f'npy_single/trues/K{i}_true.npy'))
    with open('all_pred_single.npy', 'wb') as p:
        np.save(p, np.array(predlist, dtype='<U14'))
    with open('all_true_single.npy', 'wb') as t:
        np.save(t, np.concatenate(truelist))
