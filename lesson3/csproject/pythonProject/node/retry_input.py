from graph.GraphState import GraphState


def retry_input(state: GraphState) -> GraphState:
    print("대답할 수 없는 내용입니다.")
    new_input = input("다른 질문을 입력해주세요: ")
    return GraphState(user_input=new_input)
