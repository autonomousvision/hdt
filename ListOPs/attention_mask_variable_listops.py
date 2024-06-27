import torch
import Config
import os


comparison_mask_12 = torch.tensor([
    [1,1,1],
    [1,1,1],
    [1,1,1],
])

def load_token_ids():
    import pickle
    with open(os.path.join(Config.TOKENIZER_PATH, 'listops_token_ids.pkl'), 'rb') as reader:
        token_ids = pickle.load(reader)

    charset = '[]0123456789SUMMEDMAXMIN'
    return token_ids

def compute_attn_mask(input_ids, charset_ids, attn_masks, i):
    open_bracket = charset_ids[0]
    closed_bracket = charset_ids[1]
    numbers = charset_ids[2:12]
    operators = charset_ids[12:16]
    padding = charset_ids[16]

    if input_ids[0] in numbers:
        attn_masks[i, 0, 0] = 1
        return attn_masks
    operand_idxs_stack = []

    attn_idx = 0
    for token_id in input_ids:
        if token_id == open_bracket:
            pass
        elif token_id == closed_bracket:
            operand_idxs_stack.pop()
        elif token_id in numbers:
            attn_masks[i, attn_idx, attn_idx] = 1
            attn_masks[i, operand_idxs_stack[-1], attn_idx] = 1
        elif token_id in operators: # Operand
            operand_idxs_stack.append(attn_idx)
            attn_masks[i, attn_idx, attn_idx] = 1
            if len(operand_idxs_stack) > 1:
                attn_masks[i, operand_idxs_stack[-2], attn_idx] = 1
        elif token_id == padding:
            break
        else:
            raise Exception(f"Unknown character {token_id}")
        attn_idx += 1
    return attn_masks

def compute_attn_mask_red(input_ids, charset_ids, attn_masks, i):
    open_bracket = charset_ids[0]
    closed_bracket = charset_ids[1]
    numbers = charset_ids[2:12]
    operators = charset_ids[12:16]
    padding = charset_ids[16]

    if input_ids[0] in numbers:
        attn_masks[i, 0, 0] = 1
        return attn_masks
    operand_idxs_stack = []

    attn_idx = 0
    for token_id in input_ids:
        if token_id == open_bracket:
            pass
        elif token_id == closed_bracket:
            operand_idxs_stack.pop()
        elif token_id in numbers:
            attn_masks[i, attn_idx, attn_idx] = 1
            attn_masks[i, operand_idxs_stack[-1], attn_idx] = 1
            attn_masks[i, attn_idx, operand_idxs_stack[-1]] = 1
        elif token_id in operators: # Operand
            operand_idxs_stack.append(attn_idx)
            attn_masks[i, attn_idx, attn_idx] = 1
            if len(operand_idxs_stack) > 1:
                attn_masks[i, operand_idxs_stack[-2], attn_idx] = 1
                attn_masks[i, attn_idx, operand_idxs_stack[-2]] = 1
        elif token_id == padding:
            break
        else:
            raise Exception(f"Unknown character {token_id}")
        attn_idx += 1
    return attn_masks

def compute_attn_mask_green(input_ids, charset_ids, attn_masks, i):
    open_bracket = charset_ids[0]
    closed_bracket = charset_ids[1]
    numbers = charset_ids[2:12]
    operators = charset_ids[12:16]
    padding = charset_ids[16]

    if input_ids[0] in numbers:
        attn_masks[i, 0, 0] = 1
        return attn_masks
    operand_idxs_stack = []
    current_numbers_idxs_stack = []

    attn_idx = 0
    for token_id in input_ids:
        if token_id == open_bracket:
            pass
        elif token_id == closed_bracket:
            operand_idxs_stack.pop()
            current_numbers_idxs_stack.pop()
        elif token_id in numbers:
            attn_masks[i, attn_idx, attn_idx] = 1

            attn_masks[i, operand_idxs_stack[-1], attn_idx] = 1
            attn_masks[i, attn_idx, operand_idxs_stack[-1]] = 1
            for number_idx in current_numbers_idxs_stack[-1]:
                attn_masks[i, attn_idx, number_idx] = 1
                attn_masks[i, number_idx, attn_idx] = 1
            current_numbers_idxs_stack[-1].append(attn_idx)

        elif token_id in operators: # Operand
            operand_idxs_stack.append(attn_idx)
            current_numbers_idxs_stack.append([])
            attn_masks[i, attn_idx, attn_idx] = 1
            if len(operand_idxs_stack) > 1:
                attn_masks[i, operand_idxs_stack[-2], attn_idx] = 1
                attn_masks[i, attn_idx, operand_idxs_stack[-2]] = 1
        elif token_id == padding:
            break
        else:
            raise Exception(f"Unknown character {token_id}")
        attn_idx += 1
    return attn_masks

def compute_attn_mask_T(input_ids, charset_ids, attn_masks, i):
    open_bracket = charset_ids[0]
    closed_bracket = charset_ids[1]
    numbers = charset_ids[2:12]
    operators = charset_ids[12:16]
    padding = charset_ids[16]

    if input_ids[0] in numbers:
        attn_masks[i, 0, 0] = 1
        return attn_masks
    operand_idxs_stack = []

    attn_idx = 0
    for token_id in input_ids:
        if token_id == open_bracket:
            pass
        elif token_id == closed_bracket:
            operand_idxs_stack.pop()
        elif token_id in numbers:
            attn_masks[i, attn_idx, attn_idx] = 1
            attn_masks[i, attn_idx, operand_idxs_stack[-1]] = 1
        elif token_id in operators: # Operand
            operand_idxs_stack.append(attn_idx)
            attn_masks[i, attn_idx, attn_idx] = 1
            if len(operand_idxs_stack) > 1:
                attn_masks[i, attn_idx, operand_idxs_stack[-2]] = 1
        elif token_id == padding:
            break
        else:
            raise Exception(f"Unknown character {token_id}")
        attn_idx += 1
    return attn_masks

def sparse_to_attention_mask(sparse_attn_mask_idxs):
    result = []
    for sample_idxs in sparse_attn_mask_idxs:
        attn_mask = torch.zeros((512,512))
        for row, col in sample_idxs:
            attn_mask[row, col] = 1
        result.append(attn_mask)
    return torch.stack(result)


import numpy as np
blueprint_zeros = torch.zeros(100, 512, 512, device=Config.DEVICE)

from cython_attention_mask import cython_attention_mask

def batch_get_attn_mask(batch_token_ids, charset_ids):
    attn_masks = np.zeros((len(batch_token_ids), 512, 512), dtype=np.int32) # blueprint_zeros.clone() # torch.zeros(100, 512, 512, device=Config.DEVICE)
    for i, token_ids in enumerate(batch_token_ids):
        cython_attention_mask.compute_attn_mask(token_ids.to('cpu').numpy().astype(np.int32), np.array(charset_ids).astype(np.int32), attn_masks, i)
    res = torch.tensor(attn_masks).to(Config.DEVICE)# torch.tensor(attn_masks, device=Config.DEVICE)
    return res # can use sparsity to reduce time taken to go on GPU


def batch_get_attn_mask_green(batch_token_ids, charset_ids):
    attn_masks = np.zeros((len(batch_token_ids), 512, 512), dtype=np.int32) # blueprint_zeros.clone() # torch.zeros(100, 512, 512, device=Config.DEVICE)
    for i, token_ids in enumerate(batch_token_ids):
        compute_attn_mask_green(token_ids.to('cpu').numpy().astype(np.int32), np.array(charset_ids).astype(np.int32), attn_masks, i)
    res = torch.tensor(attn_masks).to(Config.DEVICE)# torch.tensor(attn_masks, device=Config.DEVICE)
    return res # can use sparsity to reduce time taken to go on GPU

def batch_get_attn_mask_python(batch_token_ids, charset_ids):
    attn_masks = np.zeros((len(batch_token_ids), 512, 512), dtype=np.int32) # blueprint_zeros.clone() # torch.zeros(100, 512, 512, device=Config.DEVICE)
    for i, token_ids in enumerate(batch_token_ids):
        compute_attn_mask(token_ids.to('cpu').numpy().astype(np.int32), np.array(charset_ids).astype(np.int32), attn_masks, i)
    res = torch.tensor(attn_masks).to(Config.DEVICE)# torch.tensor(attn_masks, device=Config.DEVICE)
    return res # can use sparsity to reduce time taken to go on GPU


def batch_get_attn_mask_red(batch_token_ids, charset_ids):
    attn_masks = np.zeros((len(batch_token_ids), 512, 512), dtype=np.int32) # blueprint_zeros.clone() # torch.zeros(100, 512, 512, device=Config.DEVICE)
    for i, token_ids in enumerate(batch_token_ids):
        compute_attn_mask_red(token_ids.to('cpu').numpy().astype(np.int32), np.array(charset_ids).astype(np.int32), attn_masks, i)
    res = torch.tensor(attn_masks).to(Config.DEVICE)# torch.tensor(attn_masks, device=Config.DEVICE)
    return res # can use sparsity to reduce time taken to go on GPU

def batch_get_full_attn(batch_token_ids, charset_ids):
    attn_masks = torch.zeros((len(batch_token_ids), 512, 512)).to(Config.DEVICE) # blueprint_zeros.clone() # torch.zeros(100, 512, 512, device=Config.DEVICE)
    attn_masks[batch_token_ids != 36] = 1
    return attn_masks
