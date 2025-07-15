from langchain.prompts import PromptTemplate

# Template for llama3-instruct format
MARKDOWN_CORRECTION_TEMPLATE_LLAMA3 = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a grammar correction assistant. Your job is to correct only grammatical errors in the user's Markdown content.

Strictly follow these rules:
- Do **not** modify any placeholders (e.g., <<PH1>>, <<PH93>>, <<LIST3>>, <<SEP>>). Leave them **exactly as they appear**, including spacing and underscores.
- Do **not** add any spaces adjacent to placeholders.
- Do **not** remove, reword, rename, reformat, or relocate any placeholder.
- Do **not** merge or re-wrap lines.
- Do **not** alter Markdown formatting (e.g., headings, links, lists, or indentation).
- Do **not** add or remove any text content.
- Only correct grammar **within natural language sentences**, leaving structure unchanged.

If a sentence spans multiple lines or has placeholders in it, correct the grammar but preserve formatting and placeholders **as-is**.

Example:
- Original: "We use <<PH4>> to builds model likke this:<<PH17>><<PH18>>"
- Corrected: "We use <<PH4>> to build models like this:<<PH17>><<PH18_>>"

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