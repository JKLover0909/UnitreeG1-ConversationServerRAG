# CLAUDE.md - UnitreeG1-ConversationServerRAG

## Tổng quan dự án
Dự án cung cấp Server hội thoại tích hợp RAG cho Robot humanoid Unitree G1. Server nhận yêu cầu giọng nói/văn bản, truy vấn cơ sở dữ liệu FAISS, nhận diện ý định và trả về câu phản hồi kèm Intent ID để điều khiển robot.

## Cấu trúc mã nguồn chính
- `api_server.py`: API Server chính (FastAPI/Flask) xử lý các endpoint giao tiếp với Robot.
- `core_openai.py`: Mô hình RAG, tích hợp LLM & FAISS Vector DB.
- `INTENT_ID_PROTOCOL.md`: Định nghĩa danh sách các Intent ID và hành động tương ứng của Unitree G1.
- `API_SERVER_GUIDE.md`: Hướng dẫn tích hợp API.

## Hướng dẫn phát triển
- Ngôn ngữ: Python 3.10+
- Khi chỉnh sửa API hoặc thêm Intent mới, cần cập nhật tương ứng vào `INTENT_ID_PROTOCOL.md` và `API_SERVER_GUIDE.md`.
- Tránh commit file `.env`, `nohup.out` hoặc dữ liệu rác lên Git repository.
