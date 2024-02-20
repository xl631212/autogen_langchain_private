import streamlit as st
import os
import autogen
import base64
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import io
import sys
import tempfile
import openai
import multiprocessing
import autogen.agentchat.user_proxy_agent as upa


config_list = [
    {
        "model": "gpt-4",
        "api_key": st.secrets["OPENAI_API_KEY"]
    }
]

gpt4_api_key = config_list[0]["api_key"]
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]


class OutputCapture:
    def __init__(self):
        self.contents = []

    def write(self, data):
        self.contents.append(data)

    def flush(self):
        pass

    def get_output_as_string(self):
        return ''.join(self.contents)

class ExtendedUserProxyAgent(upa.UserProxyAgent):
    def __init__(self, *args, log_file="interaction_log.txt", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file

    def log_interaction(self, message):
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def get_human_input(self, *args, **kwargs):
        human_input = super().get_human_input(*args, **kwargs)
        self.log_interaction(f"Human input: {human_input}")
        return human_input
    

def build_vector_store(pdf_path, chunk_size=1000):
    loaders = [PyPDFLoader(pdf_path)]
    docs = []
    for l in loaders:
        docs.extend(l.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    docs = text_splitter.split_documents(docs)
    vectorstore = Chroma(
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings()
    )
    vectorstore.add_documents(docs)
    return vectorstore

def setup_qa_chain(vectorstore):
    qa = ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0),
        vectorstore.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    )
    return qa

def get_image_as_base64_string(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    
def answer_question(question, qa_chain):
    response = qa_chain({"question": question})
    return response["answer"]

def initiate_task(user_proxy, assistant, user_question):
    user_proxy.initiate_chat(
        assistant,
        message= user_question
            )
    
def initiate_task_process(queue, tmp_path, user_question):
    vectorstore = build_vector_store(tmp_path)
    qa = setup_qa_chain(vectorstore)
    def answer_question(question):
        response = qa({"question": question})
        return response["answer"]
    llm_config={
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list,
    "temperature": 0,
    "functions": [
        {
            "name": "answer_question",
            "description": "Answer any questions in relation to the paper",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask in relation to the paper",
                    }
                },
                "required": ["question"],
            },
        }
    ],
    }

    # create an AssistantAgent instance named "assistant"
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config=llm_config,
    )
    # create a UserProxyAgent instance named "user_proxy"
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "."},
        llm_config=llm_config,
        system_message="""Reply TERMINATE if the task has been solved at full satisfaction.
    Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
        function_map={"answer_question": answer_question}
    )

    output_capture = OutputCapture()
    sys.stdout = output_capture
    initiate_task(user_proxy, assistant, user_question)
    queue.put(output_capture.get_output_as_string())

def app():
    st.title("NexaAgent 0.0.1")
    
    # Sidebar introduction
    st.sidebar.header("About NexaAgent 0.0.1")
    st.sidebar.markdown("""
        🚀 **Introducing NexaAgent 0.0.1!** 
        A highly efficient PDF tool for all your needs.
        
        📄 Upload any PDF, no matter its size or the task type.
        
        ✅ Guaranteed accuracy, significantly reducing any discrepancies.
        
        🔧 Empowered by:
        - **AutoGen** 🛠️
        - **LangChain** 🌐
        - **chromadb** 🗄️
    """)
    image_path = "1.png"
    st.sidebar.image(image_path, use_column_width=True)


    # Create left and right columns
    col1, col2 = st.columns(2)

    with col1:
        # Upload PDF file
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_file:
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

            # User input for question
            user_question = st.text_area("Enter your task:", height=300)
            if user_question:
                with st.spinner("Fetching the answer..."):
                    # 使用进程来执行可能引发错误的代码
                    queue = multiprocessing.Queue()
                    process = multiprocessing.Process(target=initiate_task_process, args=(queue, tmp_path, user_question))
                    process.start()
                    process.join()

                    # 从队列中获取结果
                    captured_output = queue.get()
                    col2.text_area("", value=captured_output, height=600)

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    app()
