from typing import Dict, List

from dotenv import load_dotenv
from llama_index.core import PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI

load_dotenv()


llm = GoogleGenAI(model="gemini-3.5-flash-lite", temperature=0)


def run_zero_shot(context: str, query: str):
    # this function uses no examples at all only(context + query )
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
    filled_prompt = template_str.format(context_str=context, query_str=query)
    response = llm.complete(prompt=filled_prompt).text
    return response


def run_few_shot(context: str, query: str, examples: List[Dict[str, str]]):

    examples_str = "\n".join(
        f"Example {i + 1}: \nContext:\n{ex.get('context', '')}\n"
        f"Question: {ex.get('question', '')}\n"
        f"Answer: {ex.get('answer', '')}\n"
        for i, ex in enumerate(examples)
    )

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
    filled_prompt = template_str.format(
        context_str=context, query_str=query, examples_str=examples_str
    )
    response = llm.complete(prompt=filled_prompt).text
    return response


shots = [
    {
        "question": "What is the main purpose of the subject?",
        "context": "The subject aims to improve efficiency and solve common problems.",
        "answer": "The main purpose is to improve efficiency and address common problems.",
    },
    {
        "question": "What are the key features?",
        "context": "The system includes automation, data analysis, and real-time monitoring.",
        "answer": "The key features are:\n- Automation\n- Data analysis\n- Real-time monitoring",
    },
    {
        "question": "Why is this approach useful?",
        "context": "This approach reduces manual effort and improves accuracy.",
        "answer": "This approach is useful because it reduces manual effort and improves accuracy.",
    },
    {
        "question": "When was the system introduced?",
        "context": "The system improves efficiency and supports automated workflows.",
        "answer": "Not enough information.",
    },
    {
        "question": "What are the advantages and disadvantages?",
        "context": "The method saves time but requires additional computational resources.",
        "answer": "Advantages:\n- Saves time.\n\nDisadvantages:\n- Requires additional computational resources.",
    },
]

context = "Spider-Man: Brand New Day follows Peter Parker as he tries to rebuild his life while continuing to protect New York City. As new threats emerge, Spider-Man faces dangerous enemies and unexpected challenges that test his courage and intelligence. Caught between his responsibilities as a hero and his desire for a normal life, Peter must discover what it truly means to start over."
query = "What happens in spiderman brand new day"
print("Zero Shot Response: ", run_zero_shot(context=context, query=query))
print("-" * 10)
print("Few Shot Response: ", run_few_shot(context=context, query=query, examples=shots))
