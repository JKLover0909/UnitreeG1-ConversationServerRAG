# AGENTS.md - UnitreeG1-ConversationServerRAG

## Vai trò của AI Agent trong repo này
Repo này dành cho hệ thống hạ tầng trí tuệ nhân tạo và hội thoại của Robot Unitree G1. Các Agent làm việc trên codebase này cần tuân thủ các nguyên tắc sau:

### 1. Kiến trúc RAG & LLM
- Giữ vững cấu trúc truy vấn trong `core_openai.py`.
- Bảo đảm dữ liệu truy xuất qua FAISS index đạt độ trễ thấp (< 500ms) đáp ứng tương tác thời gian thực với Robot.

### 2. Định dạng phản hồi & Intent Protocol
- Mọi phản hồi trả về cho Robot phải kèm mã Intent đúng chuẩn quy định tại `INTENT_ID_PROTOCOL.md`.
- Tuyệt đối không thay đổi mã Intent hiện có mà chưa qua kiểm thử tương thích thiết bị.

### 3. Quy tắc an toàn & bảo mật
- Không lưu cứng các API Token / Secret Keys trong mã nguồn. Hãy dùng `os.getenv()`.
- Giữ file `.env` và `nohup.out` trong danh sách loại trừ `.gitignore`.
