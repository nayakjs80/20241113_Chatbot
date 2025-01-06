## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM

# from langchain.from import CachBackedEmbedding
# from langchain.embeddings import CacheBackedEmbeddings

from datetime import datetime

from typing import List

import os
import fitz  # PyMuPDF
import re

from directory import AxDirectory

## 환경변수 불러오기
from dotenv import load_dotenv, dotenv_values
load_dotenv()

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_paths : List[str]) -> List[Document]:
    documents = []

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        doc = loader.load()
        for d in doc:
            d.metadata['file_path'] = pdf_path
        documents.extend(doc)

    return documents

def pdf_to_document(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents


## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
def save_to_vector_store(documents: List[Document], ModelName:str) -> None:
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # store = LocalFileStore("./cash/")
    # # cashed_embedder = CacheBackedEmbeddings.from_byte_store(embeddings, store, namespace=embeddings.model)
    # cashed_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings=embeddings,
    #     document_embedding_cache=store,
    #     namespace=embeddings.model)

    # vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store = FAISS.from_documents(documents, embedding=cashed_embedder)
    NewFilePath = "ModelName/" + ModelName
    vector_store.save_local(NewFilePath)

    # vector_store.save_local("faiss_index")

## 5: Document를 기존 vector db 에 추가 ..
def add_data_vectordb(documents: List[Document]) -> None:
    
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # store = LocalFileStore("./cash/")

    # cashed_embedder = CacheBackedEmbeddings.from_bytes_store(
    #     underlying_embeddings=embeddings,
    #     document_embedding_cache=store,
    #     namespace=embeddings.model)

    vector_store = FAISS.from_documents(documents, embedding=cashed_embedder)

    # oldDb = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    oldDb = FAISS.load_local("faiss_index", cashed_embedder, allow_dangerous_deserialization=True)
    
    # index = faiss.IndexFlatL2(128) # 128차원 벡터를 위한 L2 거리 인덱스 생성 
    # faiss.read_index("path_to_existing_index", index)
    # oldDb = FAISS.read_index("faiss_index")

    oldDb.merge_from(vector_store)

    # oldDb.add(vector_store)
    
    # 현재 날짜를 가져와서 형식에 맞게 변환
    current_date = datetime.now().strftime("%Y%m%d_%H%M")
    db_name = f"faiss_index_{current_date}"
    
    # 업데이트된 벡터 데이터베이스 저장
    oldDb.save_local(db_name)
    oldDb.save_local("faiss_index")

############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question, db_name:str):

    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ## 벡터 DB 호출
    new_db = FAISS.load_local(db_name, cashed_embedder, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 5})

    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()

    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 문제를 요구 할 경우에는 최대한 어려운 문제로 만들어줘
    - 곧바로 응답결과를 말해줘
    - 지문이 있는 문제의 경우 지문까지 보여줘.

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")
    # model = ChatOpenAI(model="Sora")
    # model = OllamaLLM(model="llama3.1")
    # model = ChatOllama( model="llama3.1", temperature=0) # other params...

    return custom_rag_prompt | model | StrOutputParser()

############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

pdf_paths = []
temp_dir = "PDF_임시폴더"

# embedding create .. 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("./cash/")
cashed_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace=embeddings.model)
############################################################################################################################################

def main():

    # modelName = ""

    st.set_page_config("개인 챗봇", layout="wide")

    left_column, right_column = st.columns([1, 1])

    with left_column:
        st.header("개인 챗봇")

        # api_key = st.text_input("api_key 을 입력하세요", type="password")
        # if api_key:
        #     os.environ["OPENAI_API_KEY"] = api_key

        # search_files_and_folders()

        SerchDir = "./ModelName"
        dirs = AxDirectory.list_directories(SerchDir)
        # for dirs in os.walk(SerchDir):
        #     st.write(dirs.)
        # st.write(dirs)

        option = []
        index_option = 0

        for Select in dirs:
            st.write(Select)
            option.insert(index_option, Select)
            index_option += 1

        # option.insert()

        # st.write(option)

        st.selectbox("옵션을 선택하세요!", option)

        modelName = st.text_input("Model 이름을 입력하세요", placeholder="??")

        pdf_doc = st.file_uploader("PDF Uploader", type="pdf")
        button = st.button("PDF 업로드하기")

        


        if button:
            pdf_path = save_uploadedfile(pdf_doc)

    if modelName:
        if pdf_doc and button:
            # nIndex = 0
            
            # for filename in os.listdir(temp_dir):
            #     with st.spinner("PDF 문서 정리 중"):
            #         pdf_path = os.path.join(temp_dir, filename)
            #         st.text(str(nIndex) + "번째 File : " + pdf_path)
            #         nIndex = nIndex + 1
            #         pdf_paths.append(pdf_path)

            with st.spinner("PDF 문서 저장 중"):

                st.text(pdf_path + "ModelName:" + modelName)
                # pdf_document = pdf_to_documents(pdf_paths)
                pdf_document = pdf_to_document(pdf_path)
                smaller_documents = chunk_documents(pdf_document)
                # save_to_vector_store(smaller_documents)

                if os.path.isfile(modelName):
                    add_data_vectordb(smaller_documents)
                else :
                    save_to_vector_store(smaller_documents, modelName)


        # user_question = st.text_input(modelName + "이름의 DB 입니다." + "문서에 대해서 질문해 주세요", placeholder="??")
        user_question = st.text_area(modelName + "이름의 DB 입니다." + "문서에 대해서 질문해 주세요", placeholder="??", height=200)

        if user_question:
            st.write("사용자 질문 .. [" + user_question + "]")
            response, context = process_question(user_question, modelName)
            # st.write(response)
            # st.text_area(response, height=500)

    with right_column:
        if modelName:
            if user_question:
                if context:
                    for document in context:
                        with st.expander("관련 문서"):
                            st.write(document.page_content)


if __name__ == "__main__":
    main()
