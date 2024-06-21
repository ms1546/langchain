from langchain import OpenAI, LLMChain, PromptTemplate, Memory
from langchain.text_splitter import CharacterTextSplitter
import re

openai_api_key = ''
openai_llm = OpenAI(api_key=openai_api_key)

general_template = PromptTemplate(
    input_variables=["context", "question"],
    template="{context}\nYou are a helpful assistant. Answer the following question: {question}"
)

detailed_template = PromptTemplate(
    input_variables=["context", "question"],
    template="{context}\nYou are a knowledgeable assistant. Provide a detailed answer to the following question: {question}"
)

memory = Memory()

general_qa_chain = LLMChain(llm=openai_llm, prompt=general_template, memory=memory)
detailed_qa_chain = LLMChain(llm=openai_llm, prompt=detailed_template, memory=memory)

context = "This is a test to see how well the assistant can answer questions."
questions = [
    ("What is the capital of France?"),
    ("Who wrote 'To Kill a Mockingbird'?"),
    ("What is the boiling point of water?")
]

question_history = []

def categorize_question(question):
    detailed_keywords = ["boiling point", "explain", "detailed"]
    if any(keyword in question.lower() for keyword in detailed_keywords):
        return "detailed"
    return "general"

def fetch_external_data(question):
    external_data = "External data related to the question."
    return external_data

answers = []
for question in questions:
    q_type = categorize_question(question)
    if q_type == "general":
        answer = general_qa_chain.run(context=context, question=question)
    elif q_type == "detailed":
        external_data = fetch_external_data(question)
        context_with_external_data = f"{context}\nAdditional Information: {external_data}"
        answer = detailed_qa_chain.run(context=context_with_external_data, question=question)
    answers.append(answer)
    question_history.append((q_type, question, answer))

for q_type, question, answer in question_history:
    print(f"Q: {question}\nA: {answer}\n")

with open("question_history.txt", "w") as file:
    for q_type, question, answer in question_history:
        file.write(f"Type: {q_type}\nQ: {question}\nA: {answer}\n\n")
