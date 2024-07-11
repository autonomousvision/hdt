import gc
import random
from itertools import chain
from collections.abc import Mapping
import numpy as np
import torch
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)


def random_spans_noise_mask(length, mean_noise_span_length, noise_density):
    """
    A copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L230 (inception)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
    Noise mask consisting of random spans of noise tokens.
    The number of noise tokens and the number of noise spans and non-noise spans
    are determined deterministically as follows:
    num_noise_tokens = round(length * noise_density)
    num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
    Spans alternate between non-noise and noise, beginning with non-noise.
    Subject to the above restrictions, all masks are equally likely.
    Args:
        length: an int32 scalar (length of the incoming token sequence)
        noise_density: a float - approximate density of output mask
        mean_noise_span_length: a number
    Returns:
        a boolean tensor with shape [length]
    """

    orig_length = length

    num_noise_tokens = int(np.round(length * noise_density))
    # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))

    # avoid degeneracy by ensuring positive number of noise spans
    num_noise_spans = max(num_noise_spans, 1)
    num_nonnoise_tokens = length - num_noise_tokens

    # pick the lengths of the noise spans and the non-noise spans
    def _random_segmentation(num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
    nonnoise_span_lengths = _random_segmentation(
        num_nonnoise_tokens, num_noise_spans
    )

    interleaved_span_lengths = np.reshape(
        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
        [num_noise_spans * 2],
    )
    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
    span_start_indicator = np.zeros((length,), dtype=np.int8)
    span_start_indicator[span_starts] = True
    span_num = np.cumsum(span_start_indicator)
    is_noise = np.equal(span_num % 2, 1)

    return is_noise[:orig_length]


def compute_input_and_target_lengths(
        inputs_length, noise_density, mean_noise_span_length
    ):
    """
    A copy of copy from https://github.com/EleutherAI/oslo/blob/main/oslo/transformers/tasks/data_t5_pretraining.py#L76 (shits getting meta)
    This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """
    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length


    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

@dataclass
class DataCollatorForMaskedLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        mlm (`bool`, *optional*, defaults to `True`):
            Whether or not to use masked language modeling. If set to `False`, the labels are the same as the inputs
            with the padding tokens ignored (by setting them to -100). Otherwise, the labels are -100 for non-masked
            tokens and the value to predict for the masked token.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".

    <Tip>

    For best performance, this data collator should be used with a dataset having items that are dictionaries or
    BatchEncoding, with the `"special_tokens_mask"` key, as returned by a [`PreTrainedTokenizer`] or a
    [`PreTrainedTokenizerFast`] with the argument `return_special_tokens_mask=True`.

    </Tip>"""

    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    hierarchical: bool = True
    input_max_length: int = 512

    def __post_init__(self):
        if self.hierarchical:
            from src.HDT import HDTTokenizer
            self.hed_tokenizer = HDTTokenizer(self.tokenizer, self.input_max_length)

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            if "text" in examples[0]:
                if self.hierarchical:
                    outputs = []
                    for i, sample in enumerate(examples):
                        tokenized_data = self.hed_tokenizer(sample["text"])
                        outputs.append(tokenized_data)
                    batch = dict(zip(outputs[0].keys(),
                                     [torch.tensor(np.concatenate([np.expand_dims(d[key], axis=0) for d in outputs])) for
                                      key in outputs[0].keys()]))
                else:
                    batch = self.tokenizer([" ".join(list(chain.from_iterable(example["text"]))) for example in examples], padding="max_length", max_length=self.input_max_length, truncation=True, return_tensors="pt", return_special_tokens_mask=True, return_attention_mask=True)
                    batch = dict(zip(batch.keys(), [v.type(torch.int32) for v in batch.values()]))
            else:
                batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        batch["labels"] = batch["labels"].type(torch.LongTensor)
        return batch

    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.int32)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForHierarchicalUL2(DataCollatorMixin):
    """

    Data collator used for UL2

    """
    tokenizer: PreTrainedTokenizerBase
    r_denoising: bool = True
    r_probability: float = 0.25
    r_denoising_config: Tuple[Tuple] = ((3, 0.15),)
    s_denoising: bool = True
    s_probability: float = 0.5
    x_denoising: bool = True
    x_probability: float = 0.25
    x_denoising_config: Tuple[Tuple] = ((32, 0.5), (64, 0.2))
    pad_to_multiple_of: Optional[int] = None
    input_max_length: Optional[int] = 8192
    label_max_length: Optional[int] = 256
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"
    label_pad_token_id: int = -100

    def __post_init__(self):
        self.total_task = [0, 1, 2]
        task_prob = []
        task_prob.append(self.r_probability if self.r_denoising else 0.0)
        task_prob.append(self.s_probability if self.s_denoising else 0.0)
        task_prob.append(self.x_probability if self.x_denoising else 0.0)
        self.task_prob = task_prob
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id
        self.nlu_id = self.tokenizer.get_vocab()["<NLU>"]
        self.nlg_id = self.tokenizer.get_vocab()["<NLG>"]
        self.s2s_id = self.tokenizer.get_vocab()["<S2S>"]
        from src.HDT import HDTTokenizer
        self.hed_tokenizer = HDTTokenizer(self.tokenizer, self.input_max_length)

        assert sum(task_prob) == 1.0

    def assign_task_type(self, batch_size: int):
        '''
            Randomly assign S,R,X to each sentence based on weighted prob
        '''
        return random.choices(self.total_task, weights=self.task_prob, k=batch_size)


    def shift_first_token(self, batch, denoising_idx, first_id):
        shifted_input_ids = batch["input_ids"][denoising_idx].new_zeros(batch["input_ids"][denoising_idx].shape)
        shifted_input_ids[..., 1:] = batch["input_ids"][denoising_idx][..., :-1].clone()
        shifted_input_ids[..., 0] = first_id
        batch["input_ids"][denoising_idx] = shifted_input_ids

        shifted_keep_ids = batch["keep_ids_0"][denoising_idx].new_zeros(batch["keep_ids_0"][denoising_idx].shape)
        shifted_keep_ids[..., 1:] = batch["keep_ids_0"][denoising_idx][..., :-1].clone()
        shifted_keep_ids[..., 0] = 0
        batch["keep_ids_0"][denoising_idx] = shifted_keep_ids

        shifted_keep_ids = batch["keep_ids_1"][denoising_idx].new_zeros(batch["keep_ids_1"][denoising_idx].shape)
        shifted_keep_ids[..., 1:] = batch["keep_ids_1"][denoising_idx][..., :-1].clone()
        shifted_keep_ids[..., 0] = 0
        batch["keep_ids_1"][denoising_idx] = shifted_keep_ids

        shifted_keep_ids = batch["keep_ids_2"][denoising_idx].new_zeros(batch["keep_ids_2"][denoising_idx].shape)
        shifted_keep_ids[..., 1:] = batch["keep_ids_2"][denoising_idx][..., :-1].clone()
        shifted_keep_ids[..., 0] = 1
        batch["keep_ids_2"][denoising_idx] = shifted_keep_ids

        shifted_hash_ids = batch["hash_ids_0"][denoising_idx].new_zeros(batch["hash_ids_0"][denoising_idx].shape)
        shifted_hash_ids[..., 1:] = batch["hash_ids_0"][denoising_idx][..., :-1].clone()
        shifted_hash_ids[..., 0] = 0
        batch["hash_ids_0"][denoising_idx] = shifted_hash_ids

        shifted_hash_ids = batch["hash_ids_1"][denoising_idx].new_zeros(batch["hash_ids_1"][denoising_idx].shape)
        shifted_hash_ids[..., 1:] = batch["hash_ids_1"][denoising_idx][..., :-1].clone()
        shifted_hash_ids[..., 0] = 0
        batch["hash_ids_1"][denoising_idx] = shifted_hash_ids

        shifted_hash_ids = batch["hash_ids_2"][denoising_idx].new_zeros(batch["hash_ids_2"][denoising_idx].shape)
        shifted_hash_ids[..., 1:] = batch["hash_ids_2"][denoising_idx][..., :-1].clone()
        shifted_hash_ids[..., 0] = 0
        batch["hash_ids_2"][denoising_idx] = shifted_hash_ids

        shifted_position_ids = batch["position_ids_0"][denoising_idx].new_zeros(batch["position_ids_0"][denoising_idx].shape)
        shifted_position_ids[..., 1:] = batch["position_ids_0"][denoising_idx][..., :-1].clone()
        shifted_position_ids[..., 0] = 0
        batch["position_ids_0"][denoising_idx] = shifted_position_ids

        shifted_position_ids = batch["position_ids_1"][denoising_idx].new_zeros(batch["position_ids_1"][denoising_idx].shape)
        shifted_position_ids[..., 1:] = batch["position_ids_1"][denoising_idx][..., :-1].clone()
        shifted_position_ids[..., 0] = 0
        batch["position_ids_1"][denoising_idx] = shifted_position_ids

        shifted_position_ids = batch["position_ids_2"][denoising_idx].new_zeros(batch["position_ids_2"][denoising_idx].shape)
        shifted_position_ids[..., 1:] = batch["position_ids_2"][denoising_idx][..., :-1].clone()
        shifted_position_ids[..., 0] = 0
        batch["position_ids_2"][denoising_idx] = shifted_position_ids
        return batch


    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print(examples)
        task_ids = self.assign_task_type(len(examples))
        task_type = torch.tensor(task_ids)
        if isinstance(examples[0], Mapping) and "text" not in examples[0]:
            batch = self.tokenizer.pad(examples, return_tensors="pt",
                                       pad_to_multiple_of=self.pad_to_multiple_of, return_attention_mask=False)
        else:
            outputs = []
            for i, sample in enumerate(examples):
                tokenized_data = self.hed_tokenizer(sample["text"])
                outputs.append(tokenized_data)
            batch = dict(zip(outputs[0].keys(),
                            [torch.tensor(np.concatenate([np.expand_dims(d[key], axis=0) for d in outputs])) for key in outputs[0].keys()]))
        lengths = torch.tensor(
            [(np.array(e) - self.tokenizer.pad_token_id).nonzero()[0].shape[0] for e in batch["input_ids"]],
            dtype=torch.int32)
        max_length = batch['input_ids'].shape[-1]
        anchor_tokens = torch.logical_or(batch["keep_ids_1"], batch["keep_ids_2"]).numpy().astype(np.int64)
        new_batch = dict(zip(batch.keys(), [torch.zeros_like(i) for i in batch.values()]))
        new_batch.pop("special_tokens_mask", None)
        new_batch["labels"] = torch.zeros((batch['input_ids'].shape[0], self.label_max_length), dtype=torch.int32)
        _, expanded_length = batch['input_ids'].shape
        input_ids = batch["input_ids"]
        pad_mask = (input_ids != self.tokenizer.pad_token_id).numpy()
        r_denoising_idx = task_type == 0
        if r_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[r_denoising_idx]
            sub_anchor_tokens = anchor_tokens[r_denoising_idx]
            sub_pad_mask = pad_mask[r_denoising_idx]
            # union of different denoising settings
            for (mean_span, noise) in self.r_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            mask_indices = mask_indices & sub_pad_mask
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8), sub_anchor_tokens)
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8), sub_anchor_tokens, True)
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)[:, :self.label_max_length]
            diff = self.label_max_length - _labels.shape[-1]
            _labels = np.pad(_labels, [(0, 0), (0, diff)], 'constant',
                             constant_values=self.label_pad_token_id)
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0, 0), (0, diff)], 'constant', constant_values=self.pad_token_id)
            if batch.get("keep_ids_0", None) is not None:
                new_batch = self.filter_hierarchical_ids(batch, new_batch, input_ids_sentinel, r_denoising_idx)
            new_batch['input_ids'][r_denoising_idx] = torch.from_numpy(_sub_input_ids).type(batch["input_ids"].dtype)
            new_batch['labels'][r_denoising_idx] = torch.from_numpy(_labels).type(batch["input_ids"].dtype)
            new_batch = self.shift_first_token(new_batch, r_denoising_idx, self.nlu_id)
        s_denoising_idx = task_type == 1
        if s_denoising_idx.any():
            sub_input_ids = input_ids[s_denoising_idx]
            if batch.get("keep_ids_0", None) is not None:
                s_keep_ids = [batch["keep_ids_0"][s_denoising_idx], batch["keep_ids_1"][s_denoising_idx],
                              batch["keep_ids_2"][s_denoising_idx]]
                s_hash_ids = [batch["hash_ids_0"][s_denoising_idx], batch["hash_ids_1"][s_denoising_idx],
                              batch["hash_ids_2"][s_denoising_idx]]
                s_position_ids = [batch["position_ids_0"][s_denoising_idx], batch["position_ids_1"][s_denoising_idx],
                              batch["position_ids_2"][s_denoising_idx]]
            _labels = []
            _input_ids = []
            _keep_ids = [[], [], []]
            _hash_ids = [[], [], []]
            _position_ids = [[], [], []]
            for i, (input_id, len_) in enumerate(zip(sub_input_ids, lengths[s_denoising_idx])):
                split = max(len_ // 2, 2)
                split = max(split, len_ - self.label_max_length)
                diff = expanded_length - split
                _input_ids.append(F.pad(input_id[:split], (0, diff), 'constant', self.pad_token_id))
                past_seq = input_id[split:len_]
                if past_seq[-1] != self.tokenizer.eos_token_id:
                    past_seq[-1] = self.tokenizer.eos_token_id
                _labels.append(F.pad(past_seq, (0, self.label_max_length - past_seq.shape[0]), 'constant', self.label_pad_token_id))
                if batch.get("keep_ids_0", None) is not None:
                    for j, s_keep_id in enumerate(s_keep_ids):
                        _keep_ids[j].append(F.pad(s_keep_id[i, :split], (0, diff), 'constant'))

                    for j, s_hash_id in enumerate(s_hash_ids):
                        _hash_ids[j].append(F.pad(s_hash_id[i, :split], (0, diff), 'constant', value=1e9))

                    for j, s_position_id in enumerate(s_position_ids):
                        _position_ids[j].append(F.pad(s_position_id[i, :split], (0, diff), 'constant'))
            if batch.get("keep_ids_0", None) is not None:
                new_batch["keep_ids_0"][s_denoising_idx] = torch.stack(_keep_ids[0])
                new_batch["keep_ids_1"][s_denoising_idx] = torch.stack(_keep_ids[1])
                new_batch["keep_ids_2"][s_denoising_idx] = torch.stack(_keep_ids[2])
                new_batch["hash_ids_0"][s_denoising_idx] = torch.stack(_hash_ids[0])
                new_batch["hash_ids_1"][s_denoising_idx] = torch.stack(_hash_ids[1])
                new_batch["hash_ids_2"][s_denoising_idx] = torch.stack(_hash_ids[2])
                new_batch["position_ids_0"][s_denoising_idx] = torch.stack(_position_ids[0])
                new_batch["position_ids_1"][s_denoising_idx] = torch.stack(_position_ids[1])
                new_batch["position_ids_2"][s_denoising_idx] = torch.stack(_position_ids[2])

            new_batch['input_ids'][s_denoising_idx] = torch.stack(_input_ids)
            new_batch['labels'][s_denoising_idx] = torch.stack(_labels)
            new_batch = self.shift_first_token(new_batch, s_denoising_idx, self.s2s_id)
        x_denoising_idx = task_type == 2
        if x_denoising_idx.any():
            mask_indices = None
            sub_input_ids = input_ids[x_denoising_idx]
            sub_anchor_tokens = anchor_tokens[x_denoising_idx]
            sub_pad_mask = pad_mask[x_denoising_idx]
            for (mean_span, noise) in self.x_denoising_config:
                _mask_indices = np.array([
                    random_spans_noise_mask(expanded_length, mean_span, noise) for _ in range(len(sub_input_ids))
                ])
                if mask_indices is None:
                    mask_indices = _mask_indices
                else:
                    mask_indices = mask_indices | _mask_indices

            mask_indices = mask_indices & sub_pad_mask
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8), sub_anchor_tokens)
            labels_mask = ~mask_indices
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8), sub_anchor_tokens, True)
            _sub_input_ids = self.filter_input_ids(sub_input_ids, input_ids_sentinel)
            _labels = self.filter_input_ids(sub_input_ids, labels_sentinel)[:, :self.label_max_length]
            diff = self.label_max_length - _labels.shape[-1]
            _labels = np.pad(_labels, [(0, 0), (0, diff)], 'constant',
                             constant_values=self.label_pad_token_id)
            diff = max_length - _sub_input_ids.shape[-1]
            _sub_input_ids = np.pad(_sub_input_ids, [(0, 0), (0, diff)], 'constant', constant_values=self.pad_token_id)
            if batch.get("keep_ids_0", None) is not None:
                new_batch = self.filter_hierarchical_ids(batch, new_batch, input_ids_sentinel, x_denoising_idx)
            new_batch['input_ids'][x_denoising_idx] = torch.from_numpy(_sub_input_ids).type(batch["input_ids"].dtype)
            new_batch['labels'][x_denoising_idx] = torch.from_numpy(_labels).type(batch["input_ids"].dtype)
            new_batch = self.shift_first_token(new_batch, x_denoising_idx, self.nlg_id)
        gc.collect()
        return self.prepare_decoder_inputs_from_labels(new_batch)

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = []
        for row in input_ids_full:
            collapsed_id = row[row >= 0]
            diff = len(row) - len(collapsed_id)
            collapsed_id = np.pad(collapsed_id, (0, diff), 'constant', constant_values=self.pad_token_id)
            input_ids.append(collapsed_id)
        return np.array(input_ids)

    def filter_hierarchical_ids(self, batch, new_batch, sentinel_ids, denoising_idx):
        """
                Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
                This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
                """

        keep_ids = [batch["keep_ids_0"][denoising_idx], batch["keep_ids_1"][denoising_idx], batch["keep_ids_2"][denoising_idx]]
        hash_ids = [batch["hash_ids_0"][denoising_idx], batch["hash_ids_1"][denoising_idx], batch["hash_ids_2"][denoising_idx]]
        position_ids = [batch["position_ids_0"][denoising_idx], batch["position_ids_1"][denoising_idx], batch["position_ids_2"][denoising_idx]]
        # TODO: code for merge keep, hash, position ids
        # merge keep_ids
        new_keep_ids = []
        for keep_id in keep_ids:
            keep_id_full = np.where(sentinel_ids < 0, sentinel_ids, keep_id)
            new_keep_id = []
            for row in keep_id_full:
                collapsed_id = row[row >= 0]
                diff = len(row) - len(collapsed_id)
                collapsed_id = np.pad(collapsed_id, (0, diff), 'constant')
                new_keep_id.append(collapsed_id)
            new_keep_ids.append(np.array(new_keep_id, dtype=np.int32))
        assert np.sum(new_keep_ids[1]) == np.sum(keep_ids[1].numpy())
        assert np.sum(new_keep_ids[2]) == np.sum(keep_ids[2].numpy())
        # merge hash_ids
        new_hash_ids = []
        for hash_id in hash_ids:
            hash_id_full = np.where(sentinel_ids < 0, sentinel_ids, hash_id)
            new_hash_id = []
            for row in hash_id_full:
                collapsed_id = row[row >= 0]
                diff = len(row) - len(collapsed_id)
                collapsed_id = np.pad(collapsed_id, (0, diff), 'constant', constant_values=1e9)
                new_hash_id.append(collapsed_id)
            new_hash_ids.append(np.array(new_hash_id, dtype=np.int32))

        # merge position_ids
        new_position_ids = []
        for hid, position_id in enumerate(position_ids):
            position_id_full = np.where(sentinel_ids < 0, sentinel_ids, position_id)
            new_position_id = []
            for row in position_id_full:
                collapsed_id = row[row >= 0]
                diff = len(row) - len(collapsed_id)
                collapsed_id = np.pad(collapsed_id, (0, diff), 'constant')
                if hid == 0:
                    collapsed_id = enumerate_bool_values_with_reset(collapsed_id)
                new_position_id.append(collapsed_id)
            new_position_ids.append(np.array(new_position_id, dtype=np.int32))

        keep_id_type = new_batch["keep_ids_0"].dtype
        hash_id_type = new_batch["hash_ids_0"].dtype
        position_id_type = new_batch["position_ids_0"].dtype
        new_batch["keep_ids_0"][denoising_idx] = torch.tensor(new_keep_ids[0], dtype=keep_id_type)
        new_batch["keep_ids_1"][denoising_idx] = torch.tensor(new_keep_ids[1], dtype=keep_id_type)
        new_batch["keep_ids_2"][denoising_idx] = torch.tensor(new_keep_ids[2], dtype=keep_id_type)
        new_batch["hash_ids_0"][denoising_idx] = torch.tensor(new_hash_ids[0], dtype=hash_id_type)
        new_batch["hash_ids_1"][denoising_idx] = torch.tensor(new_hash_ids[1], dtype=hash_id_type)
        new_batch["hash_ids_2"][denoising_idx] = torch.tensor(new_hash_ids[2], dtype=hash_id_type)
        new_batch["position_ids_0"][denoising_idx] = torch.tensor(new_position_ids[0], dtype=position_id_type)
        new_batch["position_ids_1"][denoising_idx] = torch.tensor(new_position_ids[1], dtype=position_id_type)
        new_batch["position_ids_2"][denoising_idx] = torch.tensor(new_position_ids[2], dtype=position_id_type)
        return new_batch

    def create_sentinel_ids(self, mask_indices, anchor_tokens, label=False):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_sent_indices = (np.roll(anchor_tokens, 1, axis=-1) - anchor_tokens) > 0
        # because the first element is always <doc>, which is anchor tokens, so we don't need to take care of modifying the first element value messed by by rolling
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices
        if label:
            return np.where(np.logical_and(anchor_tokens, sentinel_ids == 0), -1, sentinel_ids)
        final_sentinel_ids = np.where(np.logical_and(start_sent_indices, sentinel_ids == -1), torch.cummin(torch.tensor(np.where(sentinel_ids <= 0, 1e8, sentinel_ids), dtype=torch.int64), dim=-1).values.numpy(), sentinel_ids)
        final_sentinel_ids = np.where(anchor_tokens, 0, final_sentinel_ids)
        # align sentinel ids if they are anchor tokens
        final_sentinel_ids = np.where(final_sentinel_ids > 0, final_sentinel_ids + (len(self.tokenizer) - np.max(final_sentinel_ids, axis=-1, keepdims=True) - 1), final_sentinel_ids)
        return final_sentinel_ids

    def prepare_decoder_inputs_from_labels(self, batch):
        # decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id.
        # See T5 docs for more information
        # due to the error RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int', we need the labels to be Long
        batch["labels"] = batch["labels"].type(torch.long)
        batch["labels"][batch["labels"] == self.pad_token_id] = self.label_pad_token_id
        shifted_labels = batch["labels"].new_zeros(batch["labels"].shape)
        shifted_labels[..., 1:] = batch["labels"][..., :-1].clone()
        shifted_labels[..., 0] = self.decoder_start_token_id  # decoder_start_token_id

        batch["decoder_input_ids"] = torch.masked_fill(
            shifted_labels,
            shifted_labels == self.label_pad_token_id,
            self.pad_token_id
        )
        batch["decoder_attention_mask"] = torch.where(
            shifted_labels == self.label_pad_token_id,
            0,
            torch.ones_like(shifted_labels),
        )
        gc.collect()
        return batch

def enumerate_bool_values_with_reset(id_list):
    # Initialize the enumeration value
    new_id_list = []
    enumeration_value = 0
    for idx in id_list:
        if idx == 0:
            enumeration_value = 0
            new_id_list.append(idx)
        elif idx == enumeration_value + 1:
            new_id_list.append(idx)
            enumeration_value = idx
        else:
            new_id_list.append(enumeration_value + 1)
            enumeration_value += 1
    return np.array(new_id_list, dtype=id_list.dtype)