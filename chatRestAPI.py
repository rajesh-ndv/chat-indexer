from flask import Flask, request, jsonify
import openai
from langchain import OpenAI
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTVectorStoreIndex, LLMPredictor, PromptHelper,ServiceContext,StorageContext, load_index_from_storage
import os

app = Flask(__name__)

query_engine = None
index = None

def load_query_engine():
    global query_engine, index
    if query_engine is None:
        openai.api_key = "magpie_secret"
        os.environ["OPENAI_API_KEY"] = "secret_mapgie"
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        # load index
        index = load_index_from_storage(storage_context)
        query_engine = index.as_query_engine()



def ask_ai(query):
    global query_engine, index
    if query_engine is None:
        load_query_engine()
    
    return query_engine.query(query).response




@app.route('/api/ask', methods=['POST'])
def ask_question():
    query = request.json['query']
    response = ask_ai(query)
    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
