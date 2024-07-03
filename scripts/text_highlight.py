import os

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.colors import HexColor
import matplotlib.pyplot as plt
import pickle


def interpolate_color(start_color, end_color, factor):
    """
    Interpolate between two RGB colors.
    """
    return tuple(int(start_color[i] + (end_color[i] - start_color[i]) * factor) for i in range(3))


def confidence_to_color(confidence, min_conf=-1.5, max_conf=1.5):
    """
    Map a confidence score to a color.
    Green for negative scores, red for positive scores.
    The darkness of the color changes according to the scores.
    """
    # Normalize confidence scores to a range [0, 1]
    # normalized_conf = (confidence - min_conf) / (max_conf - min_conf)

    if confidence < 0:
        # Interpolate from light green to dark green
        start_color = (158, 219, 143)  # light green (hex: #90EE90)
        end_color = (45, 219, 4)  # dark green (hex: #006400)
        factor = -confidence / min_conf
    else:
        # Interpolate from pink to dark red
        start_color = (255, 209, 209)  # light pink (hex: #FFB6C1)
        end_color = (255, 3, 3)  # dark red (hex: #8B0000)
        factor = confidence / max_conf

    color = interpolate_color(start_color, end_color, factor)

    # Convert color to hex format
    return HexColor('#%02x%02x%02x' % color)


def create_highlighted_pdf(words, question, answer, scores, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]
    story = [Paragraph(f"<font>Question: {question}</font>"), Paragraph(f"<font>Ground Truth: {answer}</font><br /><br />")]
    # words = text.split()

    if len(words) != len(scores):
        raise ValueError("The number of words does not match the number of scores.")

    # Build the formatted text with inline styles
    formatted_text = ""
    sec_count = 0
    for word, score in zip(words, scores):

        color = confidence_to_color(score, min(scores), max(scores))
        if word == "<sec>":
            word = "[SEC]"
            sec_count += 1
        elif word == "<doc>":
            word = "[DOC]"
        elif word == "<cls>":
            word = "[SENT]"
        else:
            if word.startswith("▁"):
                word = word.replace("▁", "")
            else:
                word = "-" + word
        if word == "[SEC]" and sec_count > 1:
            # Create a single paragraph with the formatted text
            formatted_text += "<br /><br />"
            paragraph = Paragraph(formatted_text, normal_style)
            formatted_text = ""
            story.append(paragraph)
        formatted_text += f'<font color="{color}">{word}</font> '
    if formatted_text != "":
        story.append(Paragraph(formatted_text, normal_style))
    doc.build(story)

def output_tokens_2_answer(tokens):
    res = ""
    for token in tokens:
        if token.startswith("▁"):
            res += " "
            res += token.lstrip("▁")
        else:
            res += token
    return res

if __name__ == "__main__":
    for file in os.listdir("./logs/hed_fa_attrs"):
        if not file.endswith(".pdf"):
            attr_res = pickle.load(open(os.path.join("./logs/hed_fa_attrs", file), "rb"))
            question = file.split("_")[-1]
            create_highlighted_pdf(attr_res.input_tokens, question, output_tokens_2_answer(attr_res.output_tokens), attr_res.seq_attr.cpu().numpy().tolist(), os.path.join("./logs/hed_fa_attrs", file + ".pdf"))