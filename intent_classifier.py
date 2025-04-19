import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class IntentClassifier:
    def __init__(self):
        """Load a Vietnamese SBERT model manually"""
        model_name = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Define categories
        self.categories = {
            "greeting": [
                "Xin chào", "Chào bạn", "Chào cậu", "Chào", "Hello", "Hi", "Xin chào chatbot", "Chào buổi sáng", "Chào buổi tối"
            ],
            "auto_build": [
                "Gợi ý giúp tôi một cấu hình PC chơi game",
                "PC giá 20 triệu tốt nhất là gì?",
                "Máy tính nào phù hợp để chỉnh sửa video?",
                "Gợi ý cấu hình máy tính văn phòng",
                "PC để chơi game AMD Ryzen 5 7600X, 2x16GB DDR5-6000Mhz, RTX 3070, khoảng 30 triệu",
                "Tôi muốn mua PC để học online, giá khoảng 15 triệu",
                "Máy tính để bàn nào tốt nhất cho sinh viên?",
                "Cần tư vấn cấu hình PC chuyên render video",
                "Tôi muốn build PC chơi game, giá khoảng 25 triệu",
                "PC chơi game giá rẻ, giá khoảng 10 triệu",
                "Máy tính bàn để chinh sửa video, giá khoảng 20 triệu",
                "PC chơi game 25tr với i7-9700k"
            ],
            "compatibility": [
                "Main này có chạy được Ryzen 7 không?",
                "RAM nào tương thích với Intel i5?",
                "Card đồ họa này có lắp vừa thùng máy không?",
                "Intel core i5 12700F có tương thích với main MSI B560 không?",
                "Mainboard Asus ROG Strix Z690-E Gaming WiFi hỗ trợ chuẩn RAM nào?",
                "Tản nhiệt phù hợp cho CPU Ryzen 9 5900X",
            ],
            "faq": [
                "Làm thế nào để chọn mainboard?",
                "Nguồn máy tính hãng nào tốt nhất?",
                "Cách chọn RAM cho PC",
                "Cách chọn card đồ họa",
                "Cách chọn CPU",
                "Làm sao để chọn ổ cứng SSD?",
                "Có hỗ trợ trả góp không?",
                "Cửa hàng có mấy chi nhánh?",
                "Chính sách bảo hành của cửa hàng",
                "Có nhận ship hàng không?",
            ]
        }

        # Generate embeddings for categories
        self.category_embeddings = {
            category: self.encode_sentences(questions)
            for category, questions in self.categories.items()
        }

    def encode_sentences(self, sentences):
        """Convert sentences to embeddings"""
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()  # CLS token embedding

    def classify(self, user_input):
        """Classify the user question by comparing embeddings."""
        input_embedding = self.encode_sentences([user_input])

        best_category = None
        best_score = -1

        for category, embeddings in self.category_embeddings.items():
            similarity = cosine_similarity(input_embedding, embeddings).max()
            if similarity > best_score:
                best_score = similarity
                best_category = category

        return best_category if best_score > 0.6 else "unknown"

if __name__ == "__main__":
    classifier = IntentClassifier()
    print(classifier.classify("Mainboard nào chạy được Ryzen 5?"))  # Output: compatibility
    print(classifier.classify("Gợi ý cấu hình máy tính giá rẻ"))  # Output: auto_build
    print(classifier.classify("PC để chơi game AMD Ryzen 5 7600X, 2x16GB DDR5-6000Mhz, RTX 3070, khoảng 30 triệu"))  # Output: auto_build