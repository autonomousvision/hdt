import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pickle
import os

from captum.attr import (
    FeatureAblation,
    ShapleyValues,
    ShapleyValueSampling,
    LayerIntegratedGradients,
    LLMGradientAttribution,
    TextTemplateInput,
    ProductBaselines,
)
from scripts.captum_wrappers import LLMAttribution, HierarchicalTextTokenInput
from src.HDT import HDTForConditionalGeneration, HDTTokenizer, HDTConfig
from src.utils import move_to_device
import datasets


if __name__ == "__main__":
    # data_index = 4
    device = "cuda"
    # model_name = "./logs/trained_models/hed_qasper"
    model_name = ""

    model_config = HDTConfig.from_pretrained(os.path.join(model_name, "best_model"))
    model = HDTForConditionalGeneration.from_pretrained(os.path.join(model_name, "best_model"), config=model_config)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hdt_tokenizer = HDTTokenizer(tokenizer, model.config.model_max_length)

    from src.data.preprocess import qasper_preprocess
    qasper_ds = datasets.load_dataset("allenai/qasper", split="validation")

    tokenized_data = qasper_preprocess(qasper_ds[0: 50], tokenizer, hdt_tokenizer, True)
    tokenized_data.keys()
    tokenized_data.pop("special_tokens_mask")
    tokenized_data.pop("answer_ids")
    tokenized_data.pop("answer_mask")
    ids = tokenized_data.pop("ids")
    questions = tokenized_data.pop("questions")
    all_answers = tokenized_data.pop("all_answers")
    documents = tokenized_data.pop("documents")
    types = tokenized_data.pop("types")
    tokenized_data = move_to_device({key: torch.from_numpy(value) for key, value in tokenized_data.items()}, device)
    tokenized_data["position_ids"] = [tokenized_data.pop("position_ids_0"), tokenized_data.pop("position_ids_1"),
                                      tokenized_data.pop("position_ids_2")]
    tokenized_data["keep_ids"] = [tokenized_data.pop("keep_ids_0"), tokenized_data.pop("keep_ids_1"),
                                      tokenized_data.pop("keep_ids_2")]
    tokenized_data["hash_ids"] = [tokenized_data.pop("hash_ids_0"), tokenized_data.pop("hash_ids_1"),
                                      tokenized_data.pop("hash_ids_2")]
    model.eval()

    fa = FeatureAblation(model)

    llm_attr = LLMAttribution(fa, tokenizer)
    for doc_i, (_id, question, answer, _type, document) in enumerate(zip(ids, questions, all_answers, types, documents)):
        if _type == "ABSTRACTIVE":
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output_ids = model.generate(input_ids=tokenized_data["input_ids"][doc_i: doc_i + 1],
                                                position_ids=[pos_ids[doc_i: doc_i + 1] for pos_ids in tokenized_data["position_ids"]],
                                                keep_ids=[pos_ids[doc_i: doc_i + 1] for pos_ids in tokenized_data["keep_ids"]],
                                                hash_ids=[pos_ids[doc_i: doc_i + 1] for pos_ids in
                                                          tokenized_data["hash_ids"]],
                                                max_new_tokens=15)
                responses = [tokenizer.decode(i, skip_special_tokens=True) for i in output_ids]
                print(responses[0])

            inp = HierarchicalTextTokenInput(document, tokenizer, hierarchical_tokenizer=hdt_tokenizer, query=question, device=model.device, baselines=" ")
            target = answer
            attr_res = llm_attr.attribute(inp, target=target)
            attr_res.model_outputs = responses[0]
            pickle.dump(attr_res, open(f"/HDT_DIRECTORY/scripts/hed_fa_abstractive_{_id}_{question}", "wb"))
    # fig, ax = attr_res.plot_token_attr(show=False)
    # fig.save_fig("../logs/test/attribution_figure.jpg")


