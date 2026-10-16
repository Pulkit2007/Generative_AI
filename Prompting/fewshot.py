from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from typing import List, Dict

load_dotenv()

def run_llm(context: str, query: str, examples : List[Dict[str, str]]) -> str:
    llm = GoogleGenAI(model="gemini-3.5-flash-lite", temperature=0)

    examples_str = "\n".join(
            f"Example {i+1}:\nContext:\n{ex.get('context','')}\n"
            f"Question: {ex.get('question','')}\n"
            f"Answer: {ex.get('answer','')}\n"
            for i, ex in enumerate(examples)
    )

    # Define prompt template
    template_str = (
        "You are an expert AI assistant.\n"
        "Use ONLY the provided context to answer the user's question. "
        "If the context is insufficient or does not mention the answer, reply exactly: "
        "'Not enough information.'\n\n"
        "Follow the style and reasoning illustrated by the examples.\n\n"
        "Examples:\n{examples_str}\n"
        "--- End of Examples ---\n\n"
        "Context:\n{context_str}\n\n"
        "User Question: {query_str}\n\n"
        "Answering Rules:\n"
        "1) Be concise and precise (3–6 sentences, unless the question requires more).\n"
        "2) Use bullet points for lists.\n"
        "3) At the end, include a 'Sources:' section with short snippets or filenames from the context you used.\n\n"
        "Final Answer:"
    )

    template = PromptTemplate(template_str)
    filled_prompt = template.format(context_str=context, query_str=query, examples_str = examples)

    # Get response
    response = llm.complete(prompt=filled_prompt)
    return response.text

shots = [
        {
            "context": "Positional encodings inject order information into sequences.",
            "question": "Why are positional encodings needed?",
            "answer": (
                "They give the model a sense of word order.\n"
                "- Without them, the model treats tokens as a bag of words.\n"
                "- Encodings ensure the sequence structure is preserved.\n"
                "Sources: lecture_notes.txt"
            )
        },
        {
            "context": "Multi-head attention projects queries, keys, and values into multiple subspaces.",
            "question": "What is the benefit of multi-head attention?",
            "answer": (
                "It lets the model learn from different representation subspaces.\n"
                "- Captures diverse relationships.\n"
                "- Improves contextual understanding.\n"
                "Sources: attention_paper.pdf"
            )
        },
    ]
    
context_text = (
        "Context from attention_mechanism.pdf"
        "In the attention mechanism, softmax is used on the similarity scores "
        "between queries and keys to produce attention weights."
    )
query_text = "What does softmax do in attention?"

response = run_llm(context_text, query_text, shots)
print(response)

