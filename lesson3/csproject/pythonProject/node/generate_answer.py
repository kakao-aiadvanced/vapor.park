
from graph.GraphState import GraphState
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from langchain_core.callbacks import BaseCallbackHandler


class PromptDebugHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("Debug - LLM Prompts:")
        for prompt in prompts:
            print(prompt)


callback = PromptDebugHandler()

openai_api_key = os.getenv("OPENAI_API_KEY")



## 유사한 질문&답 쌍을 바탕으로 LLM에게 답변생성하도록 하기
def generate_answer(state: GraphState) -> GraphState:

    print("----------✅ANSWER GENERATION---")

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

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

    # Answer Chain 설정
    answer_chain = (
            {
                "context": lambda x: x["context"],
                "question": lambda x: x["question"]
            }
            | answer_generation_prompt
            | llm
    )

    # 답변 생성
    generated_answer = answer_chain.invoke(
        {"context": state["similar_qa"], "question": state["user_input"]},
        config={"callbacks": [callback]}
    )


    print(f"Generated Answer: {generated_answer.content}")

    # 새로운 GraphState 반환
    return GraphState(
        user_input=state["user_input"],
        selected_key=state["selected_key"],
        json_file=state["json_file"],
        similar_qa=state["similar_qa"],
        generated_answer=generated_answer.content
    )
