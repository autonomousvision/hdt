from copy import copy
from typing import Callable, cast, Dict, List, Optional, Union
import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.lime import Lime
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import Attribution
from captum.attr._utils.interpretable_input import (
    InterpretableInput,
    TextTemplateInput,
    TextTokenInput,
)
from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    LayerIntegratedGradients,
    IntegratedGradients,
    TextTemplateInput,
    ProductBaselines,
)
from torch import nn, Tensor
import matplotlib.pyplot as plt
import numpy as np
from src.utils import move_to_device

DEFAULT_GEN_ARGS = {"max_new_tokens": 25, "do_sample": False}


class HierarchicalTextTokenInput(InterpretableInput):
    """
    TextTokenInput is an implementation of InterpretableInput for text inputs, whose
    interpretable features are the tokens of the text with respect to a given tokenizer.
    It is initiated with the string form of the input text and the corresponding
    tokenizer. Its input format to the model will be the tokenized id tensor,
    while its interpretable representation will be a binary tensor of the tokens
    whose values indicates if the token is “presence” or “absence”.

    Args:

        text (list): list of list of text strings for the model
        tokenizer (Tokenizer): tokenizer of the language model
        Hierarchical tokenizer (Tokenizer): hierarchical tokenizer of HDT
        baselines (int or str, optional): the
                baseline value for the tokens. It can be a string of the baseline token
                or an integer of the baseline token id. Common choices include unknown
                token or padding token. The default value is 0, which
                is commonly used for unknown token.
                Default: 0
        skip_tokens (List[int] or List[str], optional): the tokens to skip in the
                the input's interpretable representation. Use this argument to define
                uninterested tokens, commonly like special tokens, e.g., sos, and unk.
                It can be a list of strings of the tokens or a list of integers of the
                token ids.
                Default: None

    Examples::

        >>> text_inp = HierarchicalTextTokenInput("This is a test.", tokenizer)
        >>>
        >>> text_inp.to_tensor()
        >>> # the shape dependens on the tokenizer
        >>> # assuming it is broken into ["<s>", "This", "is", "a", "test", "."],
        >>> # torch.tensor([[1, 6]])
        >>>
        >>> text_inp.to_model_input(torch.tensor([[0, 1]]))
        >>> # torch.tensor([[1, 6]])

    """

    def __init__(
        self,
        text: list,
        tokenizer,
        hierarchical_tokenizer,
        query: None,
        baselines: Union[int, str] = 0,  # usually UNK
        skip_tokens: Union[List[int], List[str], None] = None,
        device: str = "cuda"
    ):

        tokenized_data = hierarchical_tokenizer(text, query, global_query=True)
        tokenized_data = move_to_device({key: torch.from_numpy(value).unsqueeze(0) for key, value in tokenized_data.items()}, device)
        tokenized_data.pop("special_tokens_mask", None)
        tokenized_data["position_ids"] = [tokenized_data.pop("position_ids_0"), tokenized_data.pop("position_ids_1"),
                                          tokenized_data.pop("position_ids_2")]
        tokenized_data["keep_ids"] = [tokenized_data.pop("keep_ids_0"), tokenized_data.pop("keep_ids_1"),
                                      tokenized_data.pop("keep_ids_2")]
        tokenized_data["hash_ids"] = [tokenized_data.pop("hash_ids_0"), tokenized_data.pop("hash_ids_1"),
                                      tokenized_data.pop("hash_ids_2")]

        inp_tensor = tokenized_data.pop("input_ids")
        self.kwargs = tokenized_data
        # input tensor into the model of token ids
        self.inp_tensor = inp_tensor
        # tensor of interpretable token ids
        self.itp_tensor = inp_tensor
        # interpretable mask
        self.itp_mask = None

        if skip_tokens:
            if isinstance(skip_tokens[0], str):
                skip_tokens = tokenizer.convert_tokens_to_ids(skip_tokens)
                assert isinstance(skip_tokens, list)

            skip_token_set = set(skip_tokens)
            itp_mask = torch.zeros_like(inp_tensor)
            itp_mask.map_(inp_tensor, lambda _, v: v not in skip_token_set)
            itp_mask = itp_mask.bool()

            itp_tensor = inp_tensor[itp_mask].unsqueeze(0)

            self.itp_tensor = itp_tensor
            self.itp_mask = itp_mask

        self.skip_tokens = skip_tokens

        # features values, the tokens
        self.values = tokenizer.convert_ids_to_tokens(self.itp_tensor[0].tolist())
        self.tokenizer = tokenizer
        self.hierarchical_tokenizer = hierarchical_tokenizer
        self.n_itp_features = len(self.values)

        self.baselines = (
            baselines
            if type(baselines) is int
            else tokenizer.convert_tokens_to_ids([baselines])[0]
        )

    def to_tensor(self) -> torch.Tensor:
        # return the perturbation indicator as interpretable tensor instead of token ids
        return torch.ones_like(self.itp_tensor)

    def to_model_input(self, perturbed_tensor=None) -> torch.Tensor:
        if perturbed_tensor is None:
            return self.inp_tensor

        device = perturbed_tensor.device

        perturb_mask = perturbed_tensor != 1

        # perturb_per_eval or gradient based can expand the batch dim
        expand_shape = (perturbed_tensor.size(0), -1)

        perturb_itp_tensor = self.itp_tensor.expand(*expand_shape).clone().to(device)
        perturb_itp_tensor[perturb_mask] = self.baselines

        # if no iterpretable mask, the interpretable tensor is the input tensor
        if self.itp_mask is None:
            return perturb_itp_tensor

        perturb_inp_tensor = self.inp_tensor.expand(*expand_shape).clone().to(device)
        itp_mask = self.itp_mask.expand(*expand_shape).to(device)

        perturb_inp_tensor[itp_mask] = perturb_itp_tensor.view(-1)

        return perturb_inp_tensor

    def format_attr(self, itp_attr: torch.Tensor):
        return itp_attr


class LLMAttributionResult:
    """
    Data class for the return result of LLMAttribution,
    which includes the necessary properties of the attribution.
    It also provides utilities to help present and plot the result in different forms.
    """

    def __init__(
        self,
        seq_attr: Tensor,
        token_attr: Union[Tensor, None],
        input_tokens: List[str],
        output_tokens: List[str],
        model_outputs: str = "",
    ):
        self.seq_attr = seq_attr
        self.token_attr = token_attr
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model_outputs = model_outputs

    @property
    def seq_attr_dict(self):
        return {k: v for v, k in zip(self.seq_attr.cpu().tolist(), self.input_tokens)}

    def plot_token_attr(self, show=False):
        """
        Generate a matplotlib plot for visualising the attribution
        of the output tokens.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
        """

        token_attr = self.token_attr.cpu()

        # maximum absolute attribution value
        # used as the boundary of normalization
        # always keep 0 as the mid point to differentiate pos/neg attr
        max_abs_attr_val = token_attr.abs().max().item()

        fig, ax = plt.subplots()

        # Plot the heatmap
        data = token_attr.numpy()

        fig.set_size_inches(
            max(data.shape[1] * 1.3, 6.4), max(data.shape[0] / 2.5, 4.8)
        )
        im = ax.imshow(
            data,
            vmax=max_abs_attr_val,
            vmin=-max_abs_attr_val,
            cmap="RdYlGn",
            aspect="auto",
        )

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Token Attribuiton", rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        ax.set_xticks(np.arange(data.shape[1]), labels=self.input_tokens)
        ax.set_yticks(np.arange(data.shape[0]), labels=self.output_tokens)

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "black" if 0.2 < im.norm(val) < 0.8 else "white"
                im.axes.text(
                    j,
                    i,
                    "%.4f" % val,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                )

        if show:
            plt.show()
        else:
            return fig, ax

    def plot_seq_attr(self, show=False):
        """
        Generate a matplotlib plot for visualising the attribution
        of the output sequence.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
        """

        fig, ax = plt.subplots()

        data = self.seq_attr.cpu().numpy()

        ax.set_xticks(range(data.shape[0]), labels=self.input_tokens)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # pos bar
        ax.bar(
            range(data.shape[0]), [max(v, 0) for v in data], align="center", color="g"
        )
        # neg bar
        ax.bar(
            range(data.shape[0]), [min(v, 0) for v in data], align="center", color="r"
        )

        ax.set_ylabel("Sequence Attribuiton", rotation=90, va="bottom")

        if show:
            plt.show()
        else:
            return fig, ax


class LLMAttribution(Attribution):
    """
    Attribution class for large language models. It wraps a perturbation-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm, but it will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    SUPPORTED_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
        Lime,
        KernelShap,
    )
    SUPPORTED_PER_TOKEN_ATTR_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
    )
    SUPPORTED_INPUTS = (HierarchicalTextTokenInput, )

    def __init__(
        self,
        attr_method: Attribution,
        tokenizer,
        attr_target: str = "log_prob",  # TODO: support callable attr_target
    ):
        """
        Args:
            attr_method (Attribution): Instance of a supported perturbation attribution
                    Supported methods include FeatureAblation, ShapleyValueSampling,
                    ShapleyValues, Lime, and KernelShap. Lime and KernelShap do not
                    support per-token attribution and will only return attribution
                    for the full target sequence.
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
            attr_target (str): attribute towards log probability or probability.
                    Available values ["log_prob", "prob"]
                    Default: "log_prob"
        """

        assert isinstance(
            attr_method, self.SUPPORTED_METHODS
        ), f"LLMAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method = copy(attr_method)
        self.include_per_token_attr = isinstance(
            attr_method, self.SUPPORTED_PER_TOKEN_ATTR_METHODS
        )

        self.attr_method.forward_func = self._forward_func

        # alias, we really need a model and don't support wrapper functions
        # coz we need call model.forward, model.generate, etc.
        self.model = cast(nn.Module, self.forward_func)

        self.tokenizer = tokenizer
        self.device = (
            cast(torch.device, self.model.device)
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

        assert attr_target in (
            "log_prob",
            "prob",
        ), "attr_target should be either 'log_prob' or 'prob'"
        self.attr_target = attr_target

    def _forward_func(
        self,
        perturbed_tensor,
        inp,
        target_tokens,
        _inspect_forward,
    ):
        perturbed_input = self._format_model_input(inp.to_model_input(perturbed_tensor))
        init_model_inp = perturbed_input

        model_inp = init_model_inp

        log_prob_list = []
        decoder_input_ids = torch.ones((model_inp.shape[0], 1), dtype=model_inp.dtype).to(model_inp.device) * self.tokenizer.bos_token_id
        for target_token in target_tokens:
            with torch.cuda.amp.autocast():
                output_logits = self.model.forward(
                    model_inp, decoder_input_ids=decoder_input_ids, **inp.kwargs
                )
            new_token_logits = output_logits.logits[:, -1]
            log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

            log_prob_list.append(log_probs[0][target_token].detach())

            decoder_input_ids = torch.cat(
                (decoder_input_ids, torch.tensor([[target_token]]).to(self.device)), dim=1
            )

        total_log_prob = sum(log_prob_list)
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        if self.include_per_token_attr:
            target_log_probs = torch.stack(
                [total_log_prob, *log_prob_list], dim=0
            ).unsqueeze(0)
        else:
            target_log_probs = total_log_prob
        target_probs = torch.exp(target_log_probs)

        if _inspect_forward:
            prompt = self.tokenizer.decode(init_model_inp[0])
            response = self.tokenizer.decode(target_tokens)

            # callback for externals to inspect (prompt, response, seq_prob)
            _inspect_forward(prompt, response, target_probs[0].tolist())

        return target_probs if self.attr_target != "log_prob" else target_log_probs

    def _format_model_input(self, model_input: Union[str, Tensor]):
        """
        Convert str to tokenized tensor
        to make LLMAttribution work with model inputs of both
        raw text and text token tensors
        """
        # return tensor(1, n_tokens)
        if isinstance(model_input, str):
            return self.tokenizer.encode(model_input, return_tensors="pt").to(
                self.device
            )
        return model_input.to(self.device)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        num_trials: int = 1,
        gen_args: Optional[Dict] = None,
        # internal callback hook can be used for logging
        _inspect_forward: Optional[Callable] = None,
        **kwargs,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            num_trials (int, optional): number of trials to run. Return is the average
                    attribibutions over all the trials.
                    Defaults: 1.
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_length": 25, "do_sample": False}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): Attribution result. token_attr will be None
                    if attr method is Lime or KernelShap.
        """

        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"LLMAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )

            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model.generate(model_inp, **inp.kwargs, **gen_args)
            target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            assert gen_args is None, "gen_args must be None when target is given"

            if type(target) is str:
                # exclude sos
                target_tokens = self.tokenizer.encode(target)
                target_tokens = torch.tensor(target_tokens)
            elif type(target) is torch.Tensor:
                target_tokens = target

        attr = torch.zeros(
            [
                1 + len(target_tokens) if self.include_per_token_attr else 1,
                inp.n_itp_features,
            ],
            dtype=torch.float,
            device=self.device,
        )

        for _ in range(num_trials):
            attr_input = inp.to_tensor().to(self.device)

            cur_attr = self.attr_method.attribute(
                attr_input,
                additional_forward_args=(inp, target_tokens, _inspect_forward),
                **kwargs,
            )

            # temp necessary due to FA & Shapley's different return shape of multi-task
            # FA will flatten output shape internally (n_output_token, n_itp_features)
            # Shapley will keep output shape (batch, n_output_token, n_input_features)
            cur_attr = cur_attr.reshape(attr.shape)

            attr += cur_attr

        attr = attr / num_trials

        attr = inp.format_attr(attr)

        return LLMAttributionResult(
            attr[0],
            attr[1:]
            if self.include_per_token_attr
            else None,  # shape(n_output_token, n_input_features)
            inp.values,
            self.tokenizer.convert_ids_to_tokens(target_tokens),
        )


class LLMGradientAttribution(Attribution):
    """
    Attribution class for large language models. It wraps a gradient-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm,
    with respect to the log probabilities of each
    generated token and the whole sequence. It will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    SUPPORTED_METHODS = (LayerIntegratedGradients, IntegratedGradients, )
    SUPPORTED_INPUTS = (HierarchicalTextTokenInput,)

    def __init__(
        self,
        attr_method,
        tokenizer,
    ):
        """
        Args:
            attr_method (Attribution): instance of a supported perturbation attribution
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
        """
        assert isinstance(
            attr_method, self.SUPPORTED_METHODS
        ), f"LLMGradientAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method = copy(attr_method)
        self.attr_method.forward_func = self._forward_func

        # alias, we really need a model and don't support wrapper functions
        # coz we need call model.forward, model.generate, etc.
        self.model = cast(nn.Module, self.forward_func)

        self.tokenizer = tokenizer
        self.device = (
            cast(torch.device, self.model.device)
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

    def _forward_func(
        self,
        perturbed_tensor: Tensor,
        inp: InterpretableInput,
        target_tokens: Tensor,  # 1D tensor of target token ids
        cur_target_idx: int,  # current target index
    ):
        perturbed_input = self._format_model_input(inp.to_model_input(perturbed_tensor))

        # if cur_target_idx:
        #     # the input batch size can be expanded by attr method
        #     output_token_tensor = (
        #         target_tokens[:cur_target_idx]
        #         .unsqueeze(0)
        #         .expand(perturbed_input.size(0), -1)
        #         .to(self.device)
        #     )
        #     new_input_tensor = torch.cat([perturbed_input, output_token_tensor], dim=1)
        # else:
        decoder_input_ids = torch.ones((perturbed_input.shape[0], 1 + cur_target_idx), dtype=perturbed_input.dtype)
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        decoder_input_ids[:, 1:] = target_tokens[:cur_target_idx]
        with torch.cuda.amp.autocast():
            output_logits = self.model(perturbed_input, decoder_input_ids=decoder_input_ids.to(perturbed_input.device), **inp.kwargs)

        new_token_logits = output_logits.logits[:, -1]
        log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

        target_token = target_tokens[cur_target_idx]
        token_log_probs = log_probs[..., target_token]

        # the attribution target is limited to the log probability
        return token_log_probs

    def _format_model_input(self, model_input):
        """
        Convert str to tokenized tensor
        """
        return model_input.to(self.device)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        gen_args: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_length": 25, "do_sample": False}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): attribution result
        """

        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"LLMGradAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )

            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            model_inp = self._format_model_input(inp.to_model_input())
            output_tokens = self.model.generate(model_inp, **inp.kwargs, **gen_args)
            target_tokens = output_tokens[0][model_inp.size(1) :]
        else:
            assert gen_args is None, "gen_args must be None when target is given"

            if type(target) is str:
                # exclude sos
                target_tokens = self.tokenizer.encode(target)[1:]
                target_tokens = torch.tensor(target_tokens)
            elif type(target) is torch.Tensor:
                target_tokens = target

        attr_inp = inp.to_tensor().to(self.device)

        attr_list = []
        for cur_target_idx, _ in enumerate(target_tokens):
            # attr in shape(batch_size, input+output_len, emb_dim)
            attr = self.attr_method.attribute(
                attr_inp,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    cur_target_idx,
                ),
                **kwargs,
            )
            attr = cast(Tensor, attr)

            # will have the attr for previous output tokens
            # cut to shape(batch_size, inp_len, emb_dim)
            if cur_target_idx:
                attr = attr[:, :-cur_target_idx]

            # the author of IG uses sum
            # https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L350
            attr = attr.sum(-1)

            attr_list.append(attr)

        # assume inp batch only has one instance
        # to shape(n_output_token, ...)
        attr = torch.cat(attr_list, dim=0)

        # grad attr method do not care the length of features in interpretable format
        # it attributes to all the elements of the output of the specified layer
        # so we need special handling for the inp type which don't care all the elements
        if isinstance(inp, TextTokenInput) and inp.itp_mask is not None:
            itp_mask = inp.itp_mask.to(self.device)
            itp_mask = itp_mask.expand_as(attr)
            attr = attr[itp_mask].view(attr.size(0), -1)

        # for all the gradient methods we support in this class
        # the seq attr is the sum of all the token attr if the attr_target is log_prob,
        # shape(n_input_features)
        seq_attr = attr.sum(0)

        return LLMAttributionResult(
            seq_attr,
            attr,  # shape(n_output_token, n_input_features)
            inp.values,
            self.tokenizer.convert_ids_to_tokens(target_tokens),
        )