"""RAG with vector DB in LangChain with two methods
1. Step by step: making db, embedding, ...
2. Chain: gist way using Retrieval object
3. One line code
"""

# Step by step
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chat_models import ChatOpenAI

# 0. Load data
file = "OutdoorClothingCatalog_1000.csv"
loader = CSVLoader(file_path=file)
docs = loader.load()

# 1. Load Model
llm_model = "gpt-3.5-turbo-0301"
llm = ChatOpenAI(temperture=0.0, model=llm_model)

# 2. Make Embedding Method
embeddings = OpenAIEmbeddings()

# 3. Make Database
db = DocArrayInMemorySearch.from_documnets(
    docs,
    embeddings
)

# 4. Retrieve Results
query = "Please suggest a shirt with sunblocking"
result_docs = db.similarity_search(query)

# 5. Prep the results to feed to the llm model
stuffed_results = "".join(result_docs[i].page_content for i in range(len(result_docs)))

# 6. Call the llm model to get the answer
response = llm.call_as_llm(f"{stuffed_results} Question: Please list all your \
                           shirts with sun protection in a table in markdown and summarize each one.")

# Using Retrieval object
from langchain.chat_models import ChatOpenAPI
from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import CSVLoader

"""Steps
1. Make db and define retrieval object
2. Define RetrievalQA
"""
retriever = db.as_retriever()
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
response = qa_stuff.run(query)

# One line code
from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])
response = index.query(query, llm=llm)