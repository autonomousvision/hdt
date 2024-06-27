
def compute_attn_mask(int[:] input_ids, int[:] charset_ids, int[:, :, :] attn_masks, int i):
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