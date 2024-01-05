from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import openai

from dotenv import load_dotenv

load_dotenv()

url = "https://www.rebgv.org/content/rebgv-org/news-archive/metro-vancouver-regulates-wood-burning-fireplaces-and-stoves.html"

def url2News(url):

    text_splitter = RecursiveCharacterTextSplitter(["\nArchives\n\n", "Learn about us"], chunk_size = 1000, chunk_overlap = 20, length_function = len)

   # url = "https://www.rebgv.org/content/rebgv-org/news-archive/metro-vancouver-regulates-wood-burning-fireplaces-and-stoves.html"

    loader = UnstructuredURLLoader([url])

    data = loader.load_and_split(text_splitter = text_splitter)

    return data[1:2]

llm = openai.OpenAI()

chain = load_summarize_chain(llm,chain_type="map_reduce")

doc = url2News(url)

summary = chain.run(doc)

print(summary)