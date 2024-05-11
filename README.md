# Chat-with-your-files-using-Gemini

Welcome to Chat with your files using Gemini, a Streamlit application that allows you to interactively chat with your documents using Google's Generative AI.

### Description
This application enables users to upload various types of documents such as PDFs, DOCX, TXT, and CSV files. Once uploaded, the application processes the documents to extract text and creates a conversational interface where users can ask questions related to the content of the documents. The system utilizes Google's Generative AI to provide detailed answers based on the context and content of the uploaded documents.

### Features
Upload multiple types of documents including PDFs, DOCX, TXT, and CSV files.
Extract text from uploaded documents.
Process text to create a conversational interface.
Utilize Google's Generative AI for question answering based on document content.
Display conversation history between the user and the system.
### Installation
To run this application locally, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies by running:

```pip install -r requirements.txt```

4. Set up your environment variables by creating a .env file in the project directory and adding your Google API key:

```GOOGLE_API_KEY=your_api_key_here```
##### *to get your API key you can visit:* https://makersuite.google.com/app/apikey
### Usage
1. Run the application by executing the following command in your terminal:

```streamlit run app.py```

2. Access the application through your web browser.
3. Upload your documents using the file uploader.
4. Click the "Submit & Process" button to start processing the uploaded documents.
5. Once processing is complete, ask questions related to the content of the documents using the chat input.
6. View the system's responses and conversation history in the main window.
