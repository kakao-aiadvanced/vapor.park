import os
# from langchain import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, StateGraph

from graph.GraphState import GraphState

openai_api_key = os.getenv("OPENAI_API_KEY")

def select_database(state: GraphState) -> GraphState:
    print("----------✅KEY AND JSON SELECTION---")

    user_input = state["user_input"]
    vectorStores = state["vectorStores"]

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)

    # 관련 키 선택 프롬프트
    key_selector_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """
             당신은 사용자 입력을 분석하여 관련된 주제를 선택하는 AI 어시스턴트입니다.
             주어진 키워드 목록에서 사용자 입력과 가장 관련 있는 키워드를 선택해야 합니다.
             관련 있는 키워드가 없다면 "관련 없음"이라고 답변하세요.
             """
             ),
            ("human",
             """
             사용자 입력: {user_input}

             키워드 목록:
             {keywords}

             위의 사용자 입력과 가장 관련 있는 키워드를 선택하거나, 관련 없다면 "관련 없음"이라고 답변하세요.
             """
             ),
        ]
    )

    # 키 선택 체인
    key_selector_chain = (
        {
            "user_input": lambda x: x,
            "keywords": lambda _: "\n".join([key for key, _ in vectorStores])
        }
        | key_selector_prompt
        | llm
    )

    # 키 선택 실행
    result = key_selector_chain.invoke(user_input)
    selected_key = result.content.strip()

    # JSON 파일 선택
    if selected_key == "관련 없음":
        json_file = "관련 없음"
    else:
        json_file = next((file for key, file in vectorStores if key in selected_key), "관련 없음")

    print(f"User Input: {user_input}")
    print(f"Selected Key: {selected_key}")
    print(f"Selected JSON File: {json_file}")
    return GraphState(
        user_input=user_input,
        selected_key=selected_key,
        json_file=json_file,
        process_further = selected_key != "관련 없음"
    )
