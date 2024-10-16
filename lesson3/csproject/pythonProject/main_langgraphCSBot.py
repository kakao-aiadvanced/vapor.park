# from langchain import PromptTemplate

from langgraph.graph import END, StateGraph

from graph.GraphState import GraphState
from node.check_hallucination import check_hallucination
from node.generate_answer import generate_answer
from node.retry_input import retry_input
from node.select_database import select_database
from node.retrieve_vectorStore import retrieve_vectorStore


# 벡터 스토어 정의
vectorStores = [
    ("날씨관련된 답변들", "weather.json"),
    ("메세지 전송 불가, 로그인 불가 관련된 이슈 답변들", "messageLogin.json"),
    ("현재 지원가능한 기능들 관련된 답변들, (캡처)", "availableFeature.json"),
    ("환경설정 관련 답변들, (시간)", "environmentTime.json"),
]

def check_similar_qa(state: GraphState):
    if state["similar_qa"].strip():
        return "has_similar_qa"
    else:
        return "no_similar_qa"

def check_has_key_and_json(state: GraphState):
    if state["process_further"]:
        return "relevant"
    else:
        return "not_relevant"



# Main Execution
if __name__ == "__main__":

    workflow = StateGraph(GraphState)

    # -----------------------------------------------------노드
    workflow.add_node("select_database", select_database)
    workflow.add_node("retry_input", retry_input)
    workflow.add_node("retrieve_vectorStore", retrieve_vectorStore)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("check_hallucination", check_hallucination)

    #------------------------------------------------------- 노드간선
    workflow.add_conditional_edges(
        "select_database",
        check_has_key_and_json,
        {
            "relevant": "retrieve_vectorStore",
            "not_relevant": "retry_input"
        }
    )
    workflow.add_edge("retry_input", "select_database")
    workflow.add_conditional_edges(
        "retrieve_vectorStore",
        check_similar_qa,
        {
            "has_similar_qa": "generate_answer",
            "no_similar_qa": END
        }
    )
    workflow.add_edge("generate_answer", "check_hallucination")
    workflow.add_edge("check_hallucination", END)


    #--------------------------------------------------------시작 노드 설정
    workflow.set_entry_point("select_database")
    graph = workflow.compile()





    # 사용자 입력 받기
    user_input = input("질문을 입력해주세요: ")
    result = graph.invoke(GraphState(
        user_input=user_input,
        selected_key="",
        json_file="",
        similar_qa="",
        vectorStores=vectorStores,
    ))






    #결과 출력
    print("\n\n\n\n\n============Final Result:=============")
    print(f"Selected Key: {result['selected_key']}")
    print(f"Selected JSON File: {result['json_file']}")
    if result['similar_qa']:
        print(f"Similar Q&A pairs:\n{result['similar_qa']}")
    else:
        print("No relevant Q&A pairs found.")