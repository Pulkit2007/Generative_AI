import os

from openai import OpenAI

os.environ["OPENAI_API_KEY"]

client = OpenAI()

response = client.responses.create(
    model="gpt-6-luna",
    input="provide some cool fun facts about anime"
)

print(response.output_text)