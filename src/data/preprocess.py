import numpy as np
from .qasper_funcs import text_to_instance, get_paragraphs_from_article, get_evidence_mask, \
        extract_answer_and_evidence
from itertools import chain


def qasper_preprocess(examples, tokenizer, hdt_tokenizer, global_query, for_training=True, structure=True, context="full_text", max_sec_length=32):
    outputs = []
    batch_questions = []
    batch_ids = []
    batch_answers = []
    documents = []
    batch_types = []
    for title, abstract, full_text, qas, id in zip(examples["title"], examples["abstract"], examples["full_text"],
                                                   examples["qas"], examples["id"]):
        document = get_paragraphs_from_article(title, abstract, full_text, context)
        if not structure:
            flatten_document = list(chain.from_iterable(document))
            sec = []
            doc = []
            for sent in flatten_document:
                sec.append(sent)
                if len(sec) == max_sec_length:
                    doc.append(sec)
                    sec = []
            if len(sec) > 0:
                doc.append(sec)
            document = doc
        for question, answers in zip(qas["question"], qas["answers"]):
            all_answers = []
            all_evidence = []
            all_evidence_masks = []
            for answer in answers["answer"]:
                answer, evidence, answer_type = extract_answer_and_evidence(answer)
                all_answers.append({"text": answer, "type": answer_type.name})
                all_evidence.append(evidence)
                evidence_mask = get_evidence_mask(evidence, document)
                all_evidence_masks.append(evidence_mask)
            if not global_query:
                all_evidence_masks = [[0] + i for i in all_evidence_masks]
            answers_to_yield = [x['text'] for x in all_answers] if for_training else [
                all_answers[0]['text']]
            types_to_yield = [x['type'] for x in all_answers] if for_training else [
                all_answers[0]['type']]
            evidence_masks_to_yield = all_evidence_masks if for_training else [all_evidence_masks[0]]
            evidence_to_yield = all_evidence if for_training else [all_evidence[0]]
            for answer, evidence, evidence_mask, q_type in zip(answers_to_yield, evidence_to_yield,
                                                       evidence_masks_to_yield, types_to_yield):
                if context == "question_and_evidence" and answer in ['Unanswerable', 'Yes', 'No']:
                    continue
                outputs.append(text_to_instance(
                    tokenizer,
                    hdt_tokenizer,
                    question,
                    document,
                    None,
                    answer,
                    global_query,
                ))
                batch_ids.append(id)
                batch_questions.append(question)
                batch_answers.append(answer)
                batch_types.append(q_type)
                documents.append(document)
    tokenized_data = dict(zip(outputs[0].keys(),
                              [np.concatenate([np.expand_dims(d[key], axis=0) for d in outputs]) for key in
                               outputs[0].keys()]))
    tokenized_data["ids"] = batch_ids
    tokenized_data["questions"] = batch_questions
    tokenized_data["all_answers"] = batch_answers
    tokenized_data["documents"] = documents
    tokenized_data["types"] = batch_types
    return tokenized_data