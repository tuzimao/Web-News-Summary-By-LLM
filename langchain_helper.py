from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import openai
from langchain.prompts import PromptTemplate


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

prompt_template = """我将给你一段新闻概括 请把这段新闻精确的总结成中文 用5句话完成:


"{text}"


总结:"""

prompt1 = PromptTemplate(template=prompt_template, input_variables=["text"])

chain = load_summarize_chain(llm,prompt=prompt1,verbose=True)

doc = url2News(url)

summary = chain.run(doc)

print(summary)