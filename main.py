from langchain import OpenAI, LLMChain, PromptTemplate

openai_api_key = ''

openai_llm = OpenAI(api_key=openai_api_key)
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="{context}\nYou are a helpful assistant. Answer the following question: {question}"
)
qa_chain = LLMChain(llm=openai_llm, prompt=prompt_template)

context = "This is a test to see how well the assistant can answer questions."
questions = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "What is the boiling point of water?"
]

answers = [qa_chain.run(context=context, question=question) for question in questions]

for question, answer in zip(questions, answers):
    print(f"Q: {question}\nA: {answer}")
