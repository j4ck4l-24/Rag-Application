from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings('ignore')
path = "data/books"
DB_PATH = "data/chroma"
embeddings_model = OpenAIEmbeddings()

def load_chunks(query):
    db = Chroma(persist_directory = DB_PATH, embedding_function = embeddings_model)
    results = db.similarity_search_with_relevance_scores(query , k=3)
    print(results)
    if len(results) == 0:
        print("no matching results")
        return 
    context_text = '\n---\n'.join([doc.page_content for doc , _score in results])
    return context_text

PROMPT_TEMPLATE = """
Answer the query based on the following context:
{context}

---

Answer the query based on the above context: {query}
Don't include the mention of context in the response
"""
query = input("query: ")

context = load_chunks(query)
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context = context, query = query)

model = ChatOpenAI()
response_text = model.invoke(prompt)
print(f'\n------\nResponse: {response_text.content}')