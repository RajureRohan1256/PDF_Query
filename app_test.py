
from PyPDF2 import PdfReader
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import streamlit as st

import os
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPAPI_API_KEY"] = ""

st.title('Welcome to Document GPT')
st.write(
  ''' OpenAI has released several versions of its Generative Pre-trained Transformer (GPT) models,
  including GPT-3, which is the latest version available up to that point. These models are designed to
   understand and generate human-like text based on the input they receive.
  '''
)
file_uploaded = st.file_uploader('Upload you PDF Here',type=['pdf'])

st.markdown('- You can ignore this Error!! - after the Upload code will run by itself')

st.title("Let's start QnAs")


# provide the path of  pdf file/files.
if pdfreader = PdfReader(file_uploaded) == Ture:
  #if file_uploaded == True:
  #  pdfreader = PdfReader(file_uploaded)
  
  
  from typing_extensions import Concatenate
  # read text from pdf
  raw_text = ''
  for i, page in enumerate(pdfreader.pages):
      content = page.extract_text()
      if content:
          raw_text += content
  
  # We need to split the text using Character Text Split such that it sshould not increse token size
  text_splitter = CharacterTextSplitter(
      separator = "\n",
      chunk_size = 800,
      chunk_overlap  = 200,
      length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  
  # We need to split the text using Character Text Split such that it sshould not increse token size
  text_splitter = CharacterTextSplitter(
      separator = "\n",
      chunk_size = 800,
      chunk_overlap  = 200,
      length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  
  
  
  # Download embeddings from OpenAI
  embeddings = OpenAIEmbeddings()
  
  document_search = FAISS.from_texts(texts, embeddings)
  
  from langchain.chains.question_answering import load_qa_chain
  from langchain.llms import OpenAI
  
  chain = load_qa_chain(OpenAI(), chain_type="stuff")
  
  
  ## Creating ML app
  
  #import streamlit as st
  
  
  
  #while True:
  #query1 = st.text_area(label='Your Prmpt Here', height=100, placeholder='Prompt')
  
  #query = st.text_input('Your Prompt', query)
  
  query1 = st.text_area(
    label = 'Display Prompt',
    height= 100,
    max_chars = 40,
    placeholder = "Prompt Here"
  )
  
  st.write('Your Prompt is : ', query1)
  
  docs = document_search.similarity_search(query1)
  answer1 = chain.run(input_documents=docs, question=query1)
  
  st.title('Answer:')
  st.write(answer1)
  
  #ans_button = st.button('Generate Answer', on_click=answer1, use_container_width=False)
  
  #if ans_button == True:
  #  st.write(answer1)
  
  
  
  ##ignore
  
  #generate_ans_button = st.button('Generate Answer')
  #st.write(generate_ans_button)
  #if generate_ans_button==True:
  #  st.write(query)
else:
  break
