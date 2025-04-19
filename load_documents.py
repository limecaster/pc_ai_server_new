# load_documents.py
import os
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb
import uuid

class DocumentLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_documents(self):
        """Read all .txt and .docx files from a folder"""
        texts = []
        for filename in os.listdir(self.folder_path):
            print(filename)
            file_path = os.path.join(self.folder_path, filename)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
            elif filename.endswith(".docx"):
                texts.append(docx2txt.process(file_path))
        return texts

class TextProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_texts(self, texts):
        chunks = []
        for text in texts:
            chunks.extend(self.text_splitter.split_text(text))
        return chunks

class EmbeddingModel:
    def __init__(self, model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()  # Mean pooling

class FAQDatabase:
    def __init__(self, db_path="./chroma_faq_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # Initialize or reuse collection without clearing on init
        self.faq_collection = self.chroma_client.get_or_create_collection(name="faq")

    def store_embeddings(self, chunks, embeddings):
        # Clear existing embeddings to avoid duplicates
        try:
            self.faq_collection.delete(where={})
        except Exception:
            pass
        for chunk, embedding in zip(chunks, embeddings):
            uid = str(uuid.uuid4())
            # Store chunk text directly for retrieval
            self.faq_collection.add(ids=[uid], embeddings=[embedding], documents=[chunk])

    def retrieve_answer(self, question_embedding, top_k=3):
        # Retrieve the single most relevant FAQ chunk
        results = self.faq_collection.query(
            query_embeddings=[question_embedding],
            n_results=1,
            include=["documents"]
        )
        docs = results.get("documents", [[]])[0]
        if docs:
            return docs[0]
        return "Xin lỗi, tôi không tìm thấy câu trả lời."

# Global embedder and FAQ DB cache to avoid reloading on each request
_embedder = EmbeddingModel()
_faq_db = None

def retrieve_faq_answer(question):
    global _faq_db
    if _faq_db is None:
        # load and embed FAQ documents once
        loader = DocumentLoader(folder_path="documents")
        texts = loader.load_documents()
        processor = TextProcessor()
        chunks = processor.split_texts(texts)
        embeddings = [_embedder.encode_text(c) for c in chunks]
        _faq_db = FAQDatabase()
        _faq_db.store_embeddings(chunks, embeddings)
    # encode and retrieve
    question_embedding = _embedder.encode_text(question)
    return _faq_db.retrieve_answer(question_embedding)

def main():
    # Load documents
    loader = DocumentLoader(folder_path="documents")
    faq_texts = loader.load_documents()

    # Process texts
    processor = TextProcessor()
    faq_chunks = processor.split_texts(faq_texts)

    # Encode texts
    embedder = EmbeddingModel()
    faq_embeddings = [embedder.encode_text(chunk) for chunk in faq_chunks]

    # Store embeddings in database
    faq_db = FAQDatabase()
    faq_db.store_embeddings(faq_chunks, faq_embeddings)

    # Interactive chatbot
    print("Nhập 'exit' để thoát.")
    while True:
        user_question = input("Bạn có câu hỏi gì không? ")
        if user_question.lower() == "exit":
            break
        question_embedding = embedder.encode_text(user_question)
        answer = faq_db.retrieve_answer(question_embedding)
        print(answer)

if __name__ == "__main__":
    main()
