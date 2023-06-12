from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper,ServiceContext
from langchain import OpenAI
import sys
import os
import numpy as np
from pymongo import MongoClient
import openai


def construct_index(directory_path):

    openai.api_key = "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"

    max_input_size = 4096

    num_outputs = 512

    max_chunk_overlap = 0.5

    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    #index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    index.storage_context.persist()

    return index



def init_construct_index():

    os.environ["OPENAI_API_KEY"] = "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"

    client = MongoClient('mongodb+srv://raju:raju@cluster0.m3dou.gcp.mongodb.net/?retryWrites=true&w=majority')

    db = client.chatbot

    trainingSet = db.faq_documents

    oData = ''

    for document in trainingSet.find():

        del document["_id"]

        oData+="Question: "+document["Question"]+"\n"

        oData+="Answer: "+document["Answer"]+"\n"

    text_file = open("context_data/data/Data.txt", "w")

    text_file.write(oData)

    text_file.close()

    construct_index('context_data/data')

    client.close()

init_construct_index()