
from typing_extensions import TypedDict
from typing import List, Tuple

class GraphState(TypedDict):
    user_input: str
    process_further: str
    selected_key: str
    json_file: str
    similar_qa: str
    generated_answer: str
    hallucination_result: str
    vectorStores: List[Tuple[str, str]]
