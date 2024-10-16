import os
import json
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# OpenAI API 키 설정
openai_api_key = os.getenv("OPENAI_API_KEY")

# Langchain 초기화
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(model="gpt-4o-mini")

# JSON 파일에서 질문과 답변 데이터 로드
def load_predefined_QA(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_vectorstore(questions, embeddings, filename="vectorstore.json"):
    vector_data = []
    for question, embedding in zip(questions, embeddings):
        vector_data.append({
            "question": question,
            "embedding": list(embedding)  # NumPy 배열을 리스트로 변환
        })

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(vector_data, f, ensure_ascii=False, indent=4)

# 질문 및 답변 데이터 로드
predefined_QA = load_predefined_QA('dataset.json')
qa_pairs = [(qa['question'], qa['answer']) for qa in predefined_QA]
questions, answers = zip(*qa_pairs)  # 질문과 답변 분리

# Chroma 벡터 저장소 생성
vectorstore = Chroma.from_texts(questions, embeddings)
retriever = vectorstore.as_retriever()

save_vectorstore(questions, embeddings, "vectorstore.json")

# 커스텀 프롬프트 정의

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         """
         "너는 CS봇이야. "
         "아래내용을 바탕으로 답변해줘. 없는말을 지어내지는마"
         "\n\n"
         "{context}"
         """
        ),
        ("human",
         "{question}"
        ),
    ]
)

relevanceCheckerPrompt = ChatPromptTemplate.from_messages(
    [
        ("human", """
        {question}
        
        위 질문이 아래의 질문내용과 유사해 ? 유사하면 "네"라고 답변하고 아니면 "아니오"라고 답변해줘
        
        {context}
        """),
    ]
)

hallucinationCheckerPrompt = ChatPromptTemplate.from_messages(
    [
        ("human", """
        다음 답변에 대해 확인해 주세요:
        {context}
        
        이 정보가 정확한가요? 답변의 출처는 무엇인가요?
        """),
    ]
)



# 사용자 입력
user_input = input("질문을 입력해주세요: ")

# retriever에서 관련 문서 검색
docs = retriever.invoke(user_input)

if docs:  # 검색된 문서가 있을 경우

    relevanceCheckerDocs = []
    for doc in docs:
        index = questions.index(doc.page_content)  # 질문의 인덱스를 찾습니다.
        answer = answers[index]  # 해당 질문의 답변을 가져옵니다.
        relevanceCheckerDocs.append(f"관련예상질문: {doc.page_content}")
        # print(f"\n:  {doc.page_content}")
    context = "\n\n".join(relevanceCheckerDocs)

    
     # 포맷된 프롬프트 생성
    formatted_prompt = relevanceCheckerPrompt.format(context=relevanceCheckerDocs, question=user_input)
    final_answer = llm(formatted_prompt)  # 포맷된 프롬프트로 답변 생성
    # print(formatted_prompt)



    if "네" in final_answer.content:
        print("----RelevanceCheck.........---------")
        print("\n\n\n답변에 '네'")

         # 페이지 콘텐츠에서 질문과 답변 추출
        formatted_docs = []  # 배열 초기화
        for doc in docs:
            index = questions.index(doc.page_content)  # 질문의 인덱스를 찾습니다.
            answer = answers[index]  # 해당 질문의 답변을 가져옵니다.
            formatted_docs.append(f"질문: {doc.page_content}\n답변: {answer}")
            # print(f"질문: {doc.page_content}")
        
        context = "\n\n".join(formatted_docs)
    
        #포맷된 프롬프트 생성
        formatted_prompt = prompt.format(context=context, question=user_input)
        # print("\n\n\n\n\n: ", formatted_prompt)
        final_answer = llm(formatted_prompt)  # 포맷된 프롬프트로 답변 생성
        print("\n답변:", final_answer.content)


        # 할루시네이션 체크
        print("----Haluci....---------")
        formatted_prompt = hallucinationCheckerPrompt.format(context=final_answer.content)
        print("\n\n\n\n\n:",formatted_prompt)
        final_answer = llm(formatted_prompt)
        print("\n답변:", final_answer.content)
        


        # prompt = prompt.format(context=context, question=user_input)
        # question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # final_answer = rag_chain.invoke(user_input)
        # print("\n\n\n\n\n\n\n\n\n답변:", final_answer)
        
    else:
        print("\n\n\n답변에 '아니오'")
        print(final_answer.content)

    
else:
    print("연관성없음")
