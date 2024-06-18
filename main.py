from langchain import OpenAI, LLMChain, PromptTemplate, Memory

# Initialize OpenAI with your API key
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
    ("general", "What is the capital of France?"),
    ("general", "Who wrote 'To Kill a Mockingbird'?"),
    ("detailed", "What is the boiling point of water?")
]

answers = []
for q_type, question in questions:
    if q_type == "general":
        answer = general_qa_chain.run(context=context, question=question)
    elif q_type == "detailed":
        answer = detailed_qa_chain.run(context=context, question=question)
    answers.append(answer)

for (q_type, question), answer in zip(questions, answers):
    print(f"Q: {question}\nA: {answer}")

def fetch_external_data(question):
    external_data = "External data related to the question."
    return external_data

for q_type, question in questions:
    if q_type == "detailed":
        external_data = fetch_external_data(question)
        context_with_external_data = f"{context}\nAdditional Information: {external_data}"
        answer = detailed_qa_chain.run(context=context_with_external_data, question=question)
        answers.append(answer)

for (q_type, question), answer in zip(questions, answers):
    print(f"Q: {question}\nA: {answer}")
