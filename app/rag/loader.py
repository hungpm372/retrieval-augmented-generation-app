import os

from langchain_community.document_loaders import DirectoryLoader

data_sources = os.path.join(os.path.dirname(__file__), 'data_sources')

# Load the documents from the directory
loader = DirectoryLoader(data_sources, glob='**/*.txt', show_progress=True)
