# Unitree G1 - Conversation Server RAG

Hệ thống Server hội thoại tích hợp RAG (Retrieval-Augmented Generation) cho Robot Unitree G1, xử lý nhận dạng ý định, truy vấn tri thức và trả lời theo ngữ cảnh thời gian thực.

## 🚀 Tính năng chính

- **API Server (`api_server.py`):** Server FastAPI/Flask phục vụ tiếp nhận audio/text từ Unitree G1.
- **Xử lý RAG (`core_openai.py`):** Tích hợp Vector Database (FAISS) phục vụ tra cứu tri thức nâng cao.
- **Phân loại ý định (`INTENT_ID_PROTOCOL.md`):** Quy định giao thức Intent ID cho robot thực hiện hành động.
- **Thử nghiệm local (`test_gemma4_lan.ipynb`, `Micheck_local.ipynb`):** Notebook test pipeline LLM/STT/TTS LAN.

## 📁 Cấu trúc dự án

```
UnitreeG1-ConversationServerRAG/
├── api_server.py           # Core Server API giao tiếp với Robot
├── core_openai.py          # Logic xử lý RAG & LLM backend
├── API_SERVER_GUIDE.md     # Hướng dẫn chi tiết sử dụng API Server
├── INTENT_ID_PROTOCOL.md   # Giao thức định nghĩa Intent ID & Action
├── audiocases_rep/         # File mẫu âm thanh thử nghiệm
├── faiss_index/            # Cơ sở dữ liệu Vector (FAISS)
└── test_gemma4_lan.ipynb   # Notebook thử nghiệm model local
```

## 🛠️ Cài đặt & Chạy

1. **Cài đặt môi trường:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Cấu hình môi trường (`.env`):**
   Tạo file `.env` dựa trên cấu hình mẫu và điền API Key / Server URL.
3. **Khởi chạy API Server:**
   ```bash
   python3 api_server.py
   ```

## 📚 Tài liệu liên quan

- [API Server Guide](API_SERVER_GUIDE.md)
- [Intent ID Protocol](INTENT_ID_PROTOCOL.md)
