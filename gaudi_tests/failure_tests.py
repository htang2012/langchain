"""Test Self-hosted LLMs."""
import pickle
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import SelfHostedHuggingFaceLLM, SelfHostedPipeline
import runhouse as rh


model_reqs = ["transformers"]


gpu = rh.cluster(ips=['172.28.0.3'],
                  ssh_creds={'ssh_user': 'root', 'ssh_private_key':'/root/.ssh/id_rsa'},
                  name='rh-cls')

#print("#################Restart server")
#gpu.restart_server(restart_ray=True)

#print("Exit now")
#exit(1)


def test_self_hosted_huggingface_pipeline_text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text generation model."""
    llm = SelfHostedHuggingFaceLLM(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={"n_positions": 1024},
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_text2text_generation() -> None:
    """Test valid call to self-hosted HuggingFace text2text generation model."""
    llm = SelfHostedHuggingFaceLLM(
        model_id="google/flan-t5-small",
        task="text2text-generation",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_self_hosted_huggingface_pipeline_summarization() -> None:
    """Test valid call to self-hosted HuggingFace summarization model."""
    llm = SelfHostedHuggingFaceLLM(
        model_id="facebook/bart-large-cnn",
        task="summarization",
        hardware=gpu,
        model_reqs=model_reqs,
    )
    output = llm("Say foo:")
    assert isinstance(output, str)


def load_pipeline() -> Any:
    """Load pipeline for testing."""
    model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    return pipe


def inference_fn(pipeline: Any, prompt: str, stop: Optional[List[str]] = None) -> str:
    """Inference function for testing."""
    return pipeline(prompt)[0]["generated_text"]


def test_init_with_local_pipeline() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    pipeline = load_pipeline()
    llm = SelfHostedPipeline.from_pipeline(
        pipeline=pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_init_with_pipeline_path() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    pipeline = load_pipeline()
    import runhouse as rh

    rh.blob(pickle.dumps(pipeline), path="models/pipeline.pkl").save().to(
        gpu, path="models"
    )
    llm = SelfHostedPipeline.from_pipeline(
        pipeline="models/pipeline.pkl",
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)


def test_init_with_pipeline_fn() -> None:
    """Test initialization with a self-hosted HF pipeline."""
    llm = SelfHostedPipeline(
        model_load_fn=load_pipeline,
        hardware=gpu,
        model_reqs=model_reqs,
        inference_fn=inference_fn,
    )
    output = llm("Say foo:")  # type: ignore
    assert isinstance(output, str)




"""Test self-hosted embeddings."""
from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.embeddings import (
    SelfHostedEmbeddings,
    SelfHostedHuggingFaceEmbeddings,
    SelfHostedHuggingFaceInstructEmbeddings,
)


def test_self_hosted_huggingface_embedding_documents() -> None:
    """Test self-hosted huggingface embeddings."""
    documents = ["foo bar"]
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_embedding_query() -> None:
    """Test self-hosted huggingface embeddings."""
    document = "foo bar"
    embedding = SelfHostedHuggingFaceEmbeddings(hardware=gpu)
    output = embedding.embed_query(document)
    assert len(output) == 768


def test_self_hosted_huggingface_instructor_embedding_documents() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    documents = ["foo bar"]
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 768


def test_self_hosted_huggingface_instructor_embedding_query() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    query = "foo bar"
    embedding = SelfHostedHuggingFaceInstructEmbeddings(hardware=gpu)
    output = embedding.embed_query(query)
    assert len(output) == 768


def get_pipeline() -> Any:
    """Get pipeline for testing."""
    model_id = "facebook/bart-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)


def inference_fn(pipeline: Any, prompt: str) -> Any:
    """Inference function for testing."""
    # Return last hidden state of the model
    if isinstance(prompt, list):
        return [emb[0][-1] for emb in pipeline(prompt)]
    return pipeline(prompt)[0][-1]


def test_self_hosted_embedding_documents() -> None:
    """Test self-hosted huggingface instruct embeddings."""
    documents = ["foo bar"] * 2
    embedding = SelfHostedEmbeddings(
        model_load_fn=get_pipeline, hardware=gpu, inference_fn=inference_fn
    )
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) == 50265


def test_self_hosted_embedding_query() -> None:
    """Test self-hosted custom embeddings."""
    query = "foo bar"
    embedding = SelfHostedEmbeddings(
        model_load_fn=get_pipeline, hardware=gpu, inference_fn=inference_fn
    )
    output = embedding.embed_query(query)
    assert len(output) == 50265


def main():
 
 test_self_hosted_huggingface_pipeline_text_generation()
 print("====1=======")


 #test_self_hosted_huggingface_pipeline_text2text_generation()
 #Could not run 'aten::empty_strided' with arguments from the 'HPU' backend.
 
 
 test_self_hosted_huggingface_pipeline_summarization()

 print("====3=======")
 #Could not run 'aten::empty_strided' with arguments from the 'HPU' backend.
 
 '''
 test_self_hosted_huggingface_embedding_documents()
 print("====4=======")
 test_self_hosted_huggingface_embedding_query()
 print("====5=======")
 test_self_hosted_huggingface_instructor_embedding_documents()
 print("====6=======")

 test_self_hosted_huggingface_instructor_embedding_query()
 print("====7=======")
 
 
 #print("------8 -----")
 #test_self_hosted_embedding_documents()
 #print("====8=======")
 #test_self_hosted_embedding_query()
 #print("====9=======")
'''
    
if __name__ == "__main__":
    main()

