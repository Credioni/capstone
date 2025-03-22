def get_prompt(question, context):
    context = "\n\n" + "\n\n".join(context) + "\n\n"

    return f"""<|im_start|>system
You are a helpful, accurate, and concise research assistant specialized in scientific knowledge from academic papers.
Your goal is to provide factual, well-structured answers based on the information provided.
<|im_end|>
<|im_start|>user
I need a detailed and accurate answer to a scientific question. Please use only the provided context and don't make up information.
Reference exactly one article from the provided context between the CONTEXT_START and CONTEXT_END by its number and title in square brackets.

The context contains text from multiple academic articles.
Each article begins with a number and title, followed by its abstract in the next line.

CONTEXT_START
{context}
CONTEXT_END

QUESTION:
{question}
<|im_end|>
<|im_start|>assistant
I'll answer the question based on the provided context, between the CONTEXT_START and CONTEXT_END.

<think>
Let me analyze the context carefully. Each article begins with a number and title, followed by its abstract.
I need to identify which single article best addresses the question.
I'll reference the article using its exact number and title in square brackets: "[X. Article Title]"
I'll base my answer strictly on the information in the selected article.
</think>
"""