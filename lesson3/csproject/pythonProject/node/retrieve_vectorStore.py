from graph.GraphState import GraphState
import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import BaseCallbackHandler


class PromptDebugHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("Debug - LLM Prompts:")
        for prompt in prompts:
            print(prompt)


callback = PromptDebugHandler()

openai_api_key = os.getenv("OPENAI_API_KEY")


# 특정 .json 읽어들어와서 vector스토리지로 만들고,
# retreive해와서 LLM에게 질문들이 유사한지 판단.
def retrieve_vectorStore(state: GraphState) -> GraphState:
    print("----------✅SETUP VECTORSTORE AND CHECKER---")
    json_file = state["json_file"]
    user_input = state["user_input"]

    # JSON 파일에서 질문과 답변 데이터 로드
    def load_predefined_QA(file_path):
        full_path = os.path.join('dataset', file_path)
        with open(full_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    predefined_QA = load_predefined_QA(json_file)
    qa_pairs = [(qa['question'], qa['answer']) for qa in predefined_QA]
    questions, answers = zip(*qa_pairs)  # 질문과 답변 분리

    # Vectorstore 설정
    embeddings_model = OpenAIEmbeddings()  # 또는 사용하는 임베딩 모델
    vectorstore = Chroma.from_texts(questions, embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 문서 포맷 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Vectorstore 관련성 체커 프롬프트
    vectorstore_relevance_checker_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", """
            {question} 

            위 질문이 아래의 질문내용에 유사한게있어? 유사하면 "네"라고 답변하고 아니면 "아니오"라고 답변해줘

            {context}
            """),
        ]
    )

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

    # Vector 체인 설정
    vector_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | vectorstore_relevance_checker_prompt
            | llm
    )

    # LLM에 질문 전달 및 관련성 확인
    relevance_result = vector_chain.invoke(user_input, config={"callbacks": [callback]})

    if "네" in relevance_result.content:
        # 유사한 질문-답변 쌍 찾기
        similar_docs = retriever.invoke(user_input)
        similar_qa = ""
        for doc in similar_docs:
            question = doc.page_content
            answer = answers[questions.index(question)]
            similar_qa += f"질문: {question}\n답변: {answer}\n\n"

        print(f"Vectorstore setup complete for {json_file}")
        print(f"Number of Q&A pairs loaded: {len(qa_pairs)}")
        print(f"Number of similar Q&A pairs found: {len(similar_docs)}")

        return GraphState(
            user_input=user_input,
            selected_key=state["selected_key"],
            similar_qa=similar_qa.strip()
        )
    else:
        print("No relevant Q&A pairs found. Ending the process.")
        return GraphState(
            user_input=user_input,
            selected_key=state["selected_key"],
            similar_qa=""
        )

