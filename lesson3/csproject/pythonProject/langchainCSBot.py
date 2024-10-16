
import os
import json
# from langchain import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

def format_docs(docs):
    return "\n\n".join([f"관련 예상 질문: {doc.page_content}" for doc in docs])


class PromptDebugHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("Debug - LLM Prompts:")
        for prompt in prompts:
            print(prompt)

# JSON 파일에서 질문과 답변 데이터 로드
def load_predefined_QA(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


# Load predefined Q&A
predefined_QA = load_predefined_QA('dataset.json')
qa_pairs = [(qa['question'], qa['answer']) for qa in predefined_QA]
questions, answers = zip(*qa_pairs)  # 질문과 답변 분리

# Initialize embeddings and vector store
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma.from_texts(questions, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)


callback = PromptDebugHandler()


# Define Prompts

# Relevance Checker Prompt
vectorstore_relevance_checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", """
        {question} 

        위 질문이 아래의 질문내용에 유사한게있어? 유사하면 "네"라고 답변하고 아니면 "아니오"라고 답변해줘

        {context}
        """),
    ]
)
vector_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | vectorstore_relevance_checker_prompt
        | llm
)

# Answer Generation Prompt
answer_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         너는 CS봇이야.
         아래 질문&답변들 바탕으로 human의 질문 답변해줘. 없는 말을 지어내지는 마.
         WindowOS내용은 포함하면안돼. 

         {context}
         """
         ),
        ("human",
         "{question}"
         ),
    ]
)
answer_chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"]
            }
            | answer_generation_prompt
            | llm

            )

# Hallucination Checker Prompt
hallucination_checker_prompt = ChatPromptTemplate.from_messages(
    [
        ("human",
         """
        answer: {answer} 

        위의 answer 내용이 아래의 내용의 출처들을 바탕으로 할루시네이션 없이 작성된 건가요?
        맞으면 "hallucination EXIST" 이라고 대답하고, 그렇지 않으면 "hallucination NO EXIST" 라고 대답해줘.
        {context}
        """
         ),
    ]
)

hallucination_chain = (
            {
                "answer": lambda x: x["answer"],
                "context":  lambda x: x["context"],
            }
            | hallucination_checker_prompt
            | llm
            )


def rag_chain(user_question):

    docs = retriever.invoke(user_question)

    if docs:
        print("\n\n----✅ 벡터스토어에 유사 질문이 있음....------")

        relevance_result = vector_chain.invoke(user_question, config={"callbacks": [callback]})

        if "네" in relevance_result.content:
            print("\n\n----✅ LLM이 벡터스토어 유사 질문과 유저 질문이 유사하다고 함....------")


            answer_context = "\n\n".join(
                [f"질문: {doc.page_content}\n답변: {answers[questions.index(doc.page_content)]}" for doc in docs])
            generated_answer = answer_chain.invoke({"context": answer_context, "question": user_question}, config={"callbacks": [callback]})
            print("\n\n----✅ LLM이 벡터스토어 유사 질문 바탕으로 유저 질문 답변 생성....------")


            hallucination_context = "\n\n".join([f"출처: {answers[questions.index(doc.page_content)]}" for doc in docs])
            hallucination_result = hallucination_chain.invoke({"answer": generated_answer.content, "context": hallucination_context}, config={"callbacks": [callback]})


            print("\n\n----....... LLM이 생성한 답변에 할루시네이션 있는지 체크 ....------")
            if "hallucination NO EXIST" in hallucination_result.content:
                print("\n\n----✅ LLM이 생성한 답변에 할루시네이션이 없음.....------")
                print("답변:", generated_answer.content)
            else:
                print("\n\n----❌ LLM이 생성한 답변에 할루시네이션이 있음.....------")
                print("답변:", generated_answer.content)

        else:
            print("\n\n----❌ LLM이 벡터스토어 유사 질문과 유저 질문이 유사하지 않다고 함....------")
            print(relevance_result.content)
    else:
        print("연관성 없음")


# Main Execution
if __name__ == "__main__":
    user_input = input("질문을 입력해주세요: ")
    rag_chain(user_input)