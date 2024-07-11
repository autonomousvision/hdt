import numpy as np
from enum import Enum
from nltk.tokenize import sent_tokenize
from typing import List, Tuple

class AnswerType(Enum):
    EXTRACTIVE = 1
    ABSTRACTIVE = 2
    BOOLEAN = 3
    NONE = 4


def get_paragraphs_from_article(title, abstract, full_text, context):
    if context == "question_only":
        return []
    if context == "question_and_abstract":
        return [sent_tokenize(abstract)]
    sections = [[title], sent_tokenize(abstract)]
    for section_name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        # TODO (pradeep): It is possible there are other discrepancies between plain text, LaTeX and HTML.
        # Do a thorough investigation and add tests.
        # Skip paragraphs that is without any content
        if len(paragraphs) == 1 and paragraphs[0] == "":
            continue
        section = []
        if section_name is not None:
            section.append(section_name)
        for paragraph in paragraphs:
            paragraph_text = paragraph.replace("\n", " ").strip()
            if paragraph_text:
                section.extend(sent_tokenize(paragraph_text))
        sections.append(section)
        if context == "question_and_introduction":
            # Assuming the first section is the introduction and stopping here.
            break
    return sections


def extract_answer_and_evidence(answer: List) -> Tuple[str, List[str]]:
    evidence_spans = [x.replace("\n", " ").strip() for x in answer["evidence"]]
    evidence_spans = [x for x in evidence_spans if x != ""]

    answer_string = None
    answer_type = None
    if answer.get("unanswerable", False):
        answer_string = "Unanswerable"
        answer_type = AnswerType.NONE
    elif answer.get("yes_no") is not None:
        answer_string = "Yes" if answer["yes_no"] else "No"
        answer_type = AnswerType.BOOLEAN
    elif answer.get("extractive_spans", []):
        answer_string = ", ".join(answer["extractive_spans"])
        answer_type = AnswerType.EXTRACTIVE
    else:
        answer_string = answer.get("free_form_answer", "")
        answer_type = AnswerType.ABSTRACTIVE

    return answer_string, evidence_spans, answer_type


def get_evidence_mask(evidence: List[str], paragraphs: List[List[str]]) -> List[int]:
    """
    Takes a list of evidence snippets, and the list of all the paragraphs from the
    paper, and returns a list of indices of the paragraphs that contain the evidence.
    """
    evidence_mask = []
    for paragraph in paragraphs:
        for evidence_str in evidence:
            # make sure we don't miss evidence due to sentence tokenization
            if evidence_str in " ".join(paragraph) or evidence_str in "".join(paragraph):
                evidence_mask.append(1)
                break
        else:
            evidence_mask.append(0)
    return evidence_mask


def text_to_instance(
        tokenizer,
        hierarchical_tokenizer,
        question: str,
        paragraphs: List[List[str]],
        evidence_mask: List[int] = None,
        answer: str = None,
        global_query: bool = False,
        max_length: int = 128,
):
    tokenization_outputs = hierarchical_tokenizer(paragraphs, query=question if question !="" else None, global_query=global_query)
    # fields["mask"] = TensorField(torch.tensor(tokenization_outputs["mask"]))
    # fields["query_ids"] = TensorField(torch.tensor(tokenization_outputs["query_ids"]))
    if evidence_mask is not None:
        num_paragraphs = np.sum(tokenization_outputs["keep_ids_2"]) - 1
        evidence_mask = evidence_mask[:num_paragraphs]

    if evidence_mask is not None:
        evidence_labels = np.zeros_like(tokenization_outputs["input_ids"], dtype=np.int32)
        sec_ids = (tokenization_outputs["keep_ids_2"] == 1) & (tokenization_outputs["keep_ids_1"] == 1)
        evidence_labels[sec_ids.nonzero()] = evidence_mask
        tokenization_outputs["evidence"] = evidence_labels

    if answer:
        tokenized_answer = tokenizer(answer, return_tensors="np", truncation=True, padding="max_length", add_special_tokens=True, return_token_type_ids=False, max_length=max_length)
        if tokenized_answer["input_ids"][0][0] == tokenizer.bos_token_id:
            tokenization_outputs["answer_ids"] = tokenized_answer["input_ids"][0][1:]
            tokenization_outputs["answer_mask"] = tokenized_answer["attention_mask"][0][1:]
        else:
            tokenization_outputs["answer_ids"] = tokenized_answer["input_ids"][0]
            tokenization_outputs["answer_mask"] = tokenized_answer["attention_mask"][0]

    return tokenization_outputs


def _tokenize_paragraphs(tokenizer, paragraphs: List[str], paragraph_separator=False):
    tokenized_context = []
    paragraph_start_indices = []
    for paragraph in paragraphs:
        tokenized_paragraph = tokenizer.tokenize(paragraph)
        paragraph_start_indices.append(len(tokenized_context))
        tokenized_context.extend(tokenized_paragraph)
        if paragraph_separator:
            tokenized_context.append(tokenizer.convert_tokens_to_ids([paragraph_separator])[0])
    if paragraph_separator:
        # We added the separator after every paragraph, so we remove it after the last one.
        tokenized_context = tokenized_context[:-1]
    return tokenized_context, paragraph_start_indices
