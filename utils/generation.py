# Generation functions
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

def setup_qa_pipeline():
    qa_model_name = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

def generate_answer(query, retrieved_chunks, qa_pipeline):
    context = " ".join(retrieved_chunks)
    result = qa_pipeline(question=query, context=context)
    return result['answer']