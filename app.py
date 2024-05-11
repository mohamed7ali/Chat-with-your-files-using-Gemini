# Import packages.
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import docx2txt
import pandas as pd
from dotenv import load_dotenv


# Load the environment variables from a .env file into the environment.
load_dotenv()
# Configure the Google Generative AI module with the API key retrieved from the environment variables.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Extract text from a PDF.
def get_pdf_text(pdf):
    text=""
    pdf_reader= PdfReader(pdf)
    for page in pdf_reader.pages:
        text+= page.extract_text()

    return  text


# Extract text from a CSV.
def get_csv_text(csv_docs):
    text = ""
    df = pd.read_csv(csv_docs)
    text += df.to_string()

    return text


# Split text into chunks.
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1500)
    chunks = text_splitter.split_text(text)

    return chunks


# Prepare and save a vector store based on the input text chunks for similarity search.
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Set up a conversational chain for question answering.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context and documents, make sure to provide all the details, if the answer is not in
    provided context or document just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.5)
    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# process user questions and generate a response using a conversational chain for question answering.
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    handle_userinput(user_question, response)


# manages the conversation flow and simulate a conversation between the user and the system.
def handle_userinput(user_question, response):
    st.session_state.conversation.append({'question': user_question})
    st.session_state.conversation.append({'answer': response["output_text"]})

    for i, message in enumerate(st.session_state.conversation):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message["question"]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message["answer"]), unsafe_allow_html=True)




def main():
    # Set up the GUI.
    st.set_page_config("Chat PDF")
    st.header("Chat with your files using Gemini")
    st.write(css, unsafe_allow_html=True)

    # Intialize conversation to store questions and responses.
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Take user's question or query.
    input = st.chat_input("ask me about your files!")
    # Handle it if there is any
    if input:
        user_input(input)

    # Create the sidebar components and content.
    with st.sidebar:
        st.title("Menu:")
        # Get user documents.
        docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Concatenated text from the whole documents.
                final_text = " "

                # Check documents type or extension.
                for doc in docs:
                    file_extension = os.path.splitext(doc.name)[1].lower()


                    if file_extension == ".pdf":
                        raw_text = get_pdf_text(doc)
                        final_text += raw_text
            

                    elif file_extension == ".docx":
                        raw_text = docx2txt.process(doc)
                        final_text += raw_text


                    elif file_extension == ".txt":
                        text = doc.read()
                        raw_text = "\n".join(str(text))
                        final_text += raw_text
                    

                    elif file_extension == ".csv":
                        raw_text = get_csv_text(doc)
                        final_text += raw_text


                # Create the vecctor store for the whole documents at the end.
                text_chunks = get_text_chunks(final_text)
                get_vector_store(text_chunks)
                
                # Success message at the end :)
                st.success("Done")



if __name__ == "__main__":
    main()
