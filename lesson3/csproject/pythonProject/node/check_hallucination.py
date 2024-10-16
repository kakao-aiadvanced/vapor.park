
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



## 생성한 답변을 기존의 Q&A와 비교했을때 할루시네이션 없는지를 LLM에게 체크
def check_hallucination(state: GraphState) -> GraphState:
    print("--------✅HALLUCINATION CHECK---")

    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

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

    # Hallucination Chain 설정
    hallucination_chain = (
        {
            "answer": lambda x: x["answer"],
            "context": lambda x: x["context"],
        }
        | hallucination_checker_prompt
        | llm
    )

    # Hallucination 체크 실행
    hallucination_result = hallucination_chain.invoke(
        {"answer": state["generated_answer"], "context": state["similar_qa"]},
        config={"callbacks": [callback]}
    )

    print(f"Hallucination Check Result: {hallucination_result.content}")

    if "hallucination NO EXIST" in hallucination_result.content:
        hallucination_message = "✅✅✅ 할루시네이션 없음: 생성된 답변이 제공된 정보와 일치합니다."
        print(hallucination_message)
        print(state["generated_answer"])
    else:
        hallucination_message = "⚠️⚠️⚠️ 주의: 생성된 답변에 할루시네이션이 존재할 수 있습니다."

    # 새로운 GraphState 반환
    return GraphState(
        user_input=state["user_input"],
        selected_key=state["selected_key"],
        json_file=state["json_file"],
        similar_qa=state["similar_qa"],
        generated_answer=state["generated_answer"],
        hallucination_result=hallucination_result.content
    )

