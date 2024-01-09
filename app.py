from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma

# Load

loader = TextLoader('grimms.txt')
documents = loader.load()

# Split

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Embed & Store

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Retrieve & Answer

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
query = "What are the different trades Hans encounters during his journey?"

print(qa.invoke(query))