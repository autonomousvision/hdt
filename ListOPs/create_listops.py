# https://github.com/nyu-mll/spinn/blob/master/python/spinn/data/listops/make_data.py

import random
import numpy as np

MIN = "[MIN"
MAX = "[MAX"
MED = "[MED"
FIRST = "[FIRST"
LAST = "[LAST"
SUM_MOD = "[SUM"
END = "]"

OPERATORS = [MIN, MAX, MED, SUM_MOD]  # , FIRST, LAST]
VALUES = range(10)

def generate_listops_data(VALUE_P = 0.25, MAX_ARGS = 5, MAX_DEPTH = 20, DATA_POINTS = 100):
    def generate_tree(depth):
        if depth < MAX_DEPTH:
            r = random.random()
        else:
            r = 1

        if r > VALUE_P and depth > 1:
            value = random.choice(VALUES)
            return value
        else:
            num_values = random.randint(2, MAX_ARGS)
            values = []
            for _ in range(num_values):
                values.append(generate_tree(depth + 1))

            op = random.choice(OPERATORS)
            t = (op, values[0])
            for value in values[1:]:
                t = (t, value)
            t = (t, END)
        return t

    def to_string(t, parens=True):
        if isinstance(t, str):
            return t
        elif isinstance(t, int):
            return str(t)
        else:
            if parens:
                return to_string(t[0]) + to_string(t[1]) 

    def to_value(t):
        if not isinstance(t, tuple):
            return t
        l = to_value(t[0])
        r = to_value(t[1])
        if l in OPERATORS:  # Create an unsaturated function.
            return (l, [r])
        elif r == END:  # l must be an unsaturated function.
            if l[0] == MIN:
                return min(l[1])
            elif l[0] == MAX:
                return max(l[1])
            elif l[0] == FIRST:
                return l[1][0]
            elif l[0] == LAST:
                return l[1][-1]
            elif l[0] == MED:
                return int(np.median(l[1]))
            elif l[0] == SUM_MOD:
                return (np.sum(l[1]) % 10)
        elif isinstance(l, tuple):  # We've hit an unsaturated function and an argument.
            return (l[0], l[1] + [r])

    data = set()
    while len(data) < DATA_POINTS:
        t = generate_tree(1)
        text= to_string(t)
        length = len(text.replace("MAX", 'O').replace("MIN", 'O').replace("MED", 'O').replace("SUM", 'O'))
        if length > 512:
            continue
        data.add(t)

    df_data = []
    for example in data:
        df_data.append({'label': to_value(example), 'text': to_string(example)})
    print(df_data[-5:])
    return datasets.Dataset.from_list(df_data)

import datasets
import os
def create_dataset(VALUE_P, MAX_ARGS, MAX_DEPTH, train_size, test_size):
    dataset_save_name = f'ORIGINAL_LISTOPS_{str(VALUE_P).replace(".", "")}_{MAX_ARGS}_{MAX_DEPTH}_{train_size}_{test_size}'
    if os.path.exists(f'data/{dataset_save_name}'):
        print(f'data/listops_{dataset_save_name} already exists')
        return dataset_save_name

    huggingface_dataset = generate_listops_data(VALUE_P, MAX_ARGS, MAX_DEPTH, DATA_POINTS=train_size+test_size)
    train_dataset = huggingface_dataset.select(range(train_size))
    test_dataset = huggingface_dataset.select(range(train_size, train_size+test_size))

    save_path = f'data/{dataset_save_name}/train'
    os.makedirs(save_path, exist_ok=True)
    train_dataset.save_to_disk(save_path)

    save_path = f'data/{dataset_save_name}/test'
    os.makedirs(save_path, exist_ok=True)
    test_dataset.save_to_disk(save_path)

    return dataset_save_name

if __name__=='__main__':
    dataset = create_dataset(0.25, 2, 2, 2, 2)
    # dat = generate_listops_data(VALUE_P=0.25, MAX_ARGS=2, MAX_DEPTH=2, DATA_POINTS=5)
    a=  1
    # Maybe make attention mask here, or make it just after tokenization