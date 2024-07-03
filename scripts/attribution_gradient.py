import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pickle
import os

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    ShapleyValueSampling,
    IntegratedGradients,
    TextTemplateInput,
    ProductBaselines,
)
from scripts.captum_wrappers import LLMGradientAttribution, HierarchicalTextTokenInput
from src.HDT import HDTForConditionalGeneration, HDTTokenizer, HDTConfig
from src.utils import move_to_device
import datasets


if __name__ == "__main__":
    data_index = 4
    max_length = 512
    device = "cuda"
    # model_name = "./logs/trained_models/hed_qasper"
    model_name = "/home/haoyu/code/academic-budget-LMs/logs/lm/runs/2024-05-22_16-31-42/saved_model/"

    model_config = HDTConfig.from_pretrained(os.path.join(model_name, "best_model"))
    model_config.model_max_length = max_length
    model = HDTForConditionalGeneration.from_pretrained(os.path.join(model_name, "best_model"), config=model_config)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hdt_tokenizer = HDTTokenizer(tokenizer, model.config.model_max_length)

    from src.data.preprocess import qasper_preprocess
    qasper_ds = datasets.load_dataset("allenai/qasper", split="validation")

    tokenized_data = qasper_preprocess(qasper_ds[170:171], tokenizer, hdt_tokenizer, True)
    tokenized_data.keys()
    tokenized_data.pop("special_tokens_mask")
    tokenized_data.pop("answer_ids")
    tokenized_data.pop("answer_mask")
    ids = tokenized_data.pop("ids")
    questions = tokenized_data.pop("questions")
    all_answers = tokenized_data.pop("all_answers")
    documents = tokenized_data.pop("documents")
    tokenized_data = move_to_device({key: torch.from_numpy(value) for key, value in tokenized_data.items()}, device)
    tokenized_data["position_ids"] = [tokenized_data.pop("position_ids_0"), tokenized_data.pop("position_ids_1"),
                                      tokenized_data.pop("position_ids_2")]
    tokenized_data["keep_ids"] = [tokenized_data.pop("keep_ids_0"), tokenized_data.pop("keep_ids_1"),
                                      tokenized_data.pop("keep_ids_2")]
    tokenized_data["hash_ids"] = [tokenized_data.pop("hash_ids_0"), tokenized_data.pop("hash_ids_1"),
                                      tokenized_data.pop("hash_ids_2")]
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            output_ids = model.generate(**tokenized_data, max_new_tokens=15)
        responses = [tokenizer.decode(i, skip_special_tokens=True) for i in output_ids]
        print(responses[0])

    fa = IntegratedGradients(model)

    llm_attr = LLMGradientAttribution(fa, tokenizer)
    inp = HierarchicalTextTokenInput(documents[data_index], tokenizer, hierarchical_tokenizer=hdt_tokenizer, query=questions[data_index], device=model.device, baselines=" ")

    target = all_answers[data_index]

    attr_res = llm_attr.attribute(inp, target=target)
    pickle.dump(attr_res, open("logs/hed_shapley_sampling", "wb"))
    # fig, ax = attr_res.plot_token_attr(show=False)
    # fig.save_fig("../logs/test/attribution_figure.jpg")


