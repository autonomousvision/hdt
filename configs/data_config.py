def _set_args(tok, max_input_length, mlm_p):
    global tok_name, model_max_length, mlm_probability
    tok_name = tok
    model_max_length = max_input_length
    mlm_probability = mlm_p

num_proc = 16
max_entries_in_raw_dataset = 1e10
preprocess_batch_size = 2048
tok_name = "google-t5/t5-base"
model_max_length = 8192
vocab_size = 32768
mlm_probability = 0.15
ds_info = [{"path": "howey/unarXive", "name": "default", "split": "train"}, {"path": "howey/wiki_en", "name": "default", "split": "train"}]