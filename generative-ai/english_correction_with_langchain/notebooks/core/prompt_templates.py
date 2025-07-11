from langchain.prompts import PromptTemplate

# Template for llama3-instruct format
MARKDOWN_CORRECTION_TEMPLATE_LLAMA3 = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grammar correction assistant. Your job is to correct only grammatical errors in the user's Markdown content.

Strictly follow these rules:
- Do **not** modify any placeholders (e.g., __PLACEHOLDER_1__, __PLACEHOLDER2_42__, __LIST_BULLET_3__). Leave them **exactly as they appear**, including spacing and underscores.
- Do **not** add any spaces adjacent to placeholders.
- Do **not** remove, reword, rename, reformat, or relocate any placeholder.
- Do **not** merge or re-wrap lines.
- Do **not** alter Markdown formatting (e.g., headings, links, lists, or indentation).
- Do **not** add or remove any text content.
- Only correct grammar **within natural language sentences**, leaving structure unchanged.

If a sentence spans multiple lines or has placeholders in it, correct the grammar but preserve formatting and placeholders **as-is**.

Example:
- Original: "We use <<__PLACEHOLDER_4__>> to builds model likke this:<<__PLACEHOLDER_17__>><<__PLACEHOLDER_18__>>"
- Corrected: "We use <<__PLACEHOLDER_4__>> to build models like this:<<__PLACEHOLDER_17__>><<__PLACEHOLDER_18__>>"

The placeholder stays exactly the same with no additional spaces — only grammar is corrected.

Respond only with the corrected Markdown content. Do not explain anything.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text to correct:
{markdown}

Corrected text:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def get_markdown_correction_prompt() -> PromptTemplate:
    """
    Get the markdown correction prompt formatted for LLaMA 3 instruct.

    Returns:
        PromptTemplate: Ready to use in LangChain with LLaMA 3 format.
    """
    return PromptTemplate.from_template(MARKDOWN_CORRECTION_TEMPLATE_LLAMA3)