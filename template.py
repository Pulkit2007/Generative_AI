from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def run_llm(context: str, query: str) -> str:
    llm = GoogleGenAI(model="gemini-3.5-flash-lite", temperature=0)

    # Define prompt template
    template_str = (
        "You are an expert AI assistant.\n"
        "Use ONLY use the provided context to answer the user's question. "
        "If the context is insufficient or does not mention the answer, reply exactly: "
        "'Not enough information.'\n\n"
        "Context:\n{context_str}\n\n"
        "User Question: {query_str}\n\n"
        "Answering Rules:\n"
        "1) Be concise and precise (3–6 sentences, unless the question requires more).\n"
        "2) Use bullet points for lists.\n"
        "3) At the end, include a 'Sources:' section with short snippets or filenames from the context you used.\n\n"
        "Final Answer:"
    )

    template = PromptTemplate(template_str)
    filled_prompt = template.format(context_str=context, query_str=query)

    # Get response
    response = llm.complete(prompt=filled_prompt)
    return response.text

sample_context = (
        "Transformers use a self-attention mechanism that lets each token attend "
        "to every other token in the sequence. This enables modeling of long-range "
        "dependencies without recurrence. Positional encodings inject order "
        "information, and multi-head attention captures diverse relations.\n\n"
        "The encoder stacks layers of self-attention and feed-forward networks to "
        "build contextual representations. The decoder uses masked self-attention "
        "to maintain causality and cross-attention to consult encoder outputs."
    )

#This is zero shot prompting only(context + query)
sample_query = "How do Transformers handle long-range dependencies?"
output = run_llm(sample_context, sample_query)
print(output)