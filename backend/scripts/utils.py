from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=384, #maxlen for GLINER
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text])