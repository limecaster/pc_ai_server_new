# load_documents.py
import os
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoModel, AutoTokenizer
import torch
import chromadb

class DocumentLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_documents(self):
        """Read all .txt and .docx files from a folder"""
        texts = []
        for filename in os.listdir(self.folder_path):
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
        self.faq_collection = self.chroma_client.get_or_create_collection(name="faq")

    def store_embeddings(self, chunks, embeddings):
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.faq_collection.add(ids=[str(i)], embeddings=[embedding], metadatas=[{"text": chunk}])

    def retrieve_answer(self, question_embedding, top_k=1):
        results = self.faq_collection.query(query_embeddings=[question_embedding], n_results=top_k)
        if results["metadatas"]:
            return results["metadatas"][0][0]["text"]
        else:
            return "Xin lỗi, tôi không tìm thấy câu trả lời."

def retrieve_faq_answer(question):
    embedder = EmbeddingModel()
    question_embedding = embedder.encode_text(question)
    faq_db = FAQDatabase()
    return faq_db.retrieve_answer(question_embedding)

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
