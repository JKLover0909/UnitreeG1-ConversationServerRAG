# QUICK START - API SERVER
# =====================================================

## 🚀 Khởi động nhanh trên PC (Windows)

### Bước 1: Cài đặt dependencies

```powershell
# Cài đặt packages
pip install -r requirements_api_server.txt
```

### Bước 2: Set API Key

```powershell
# Temporary (chỉ trong terminal hiện tại)
$env:OPENAI_API_KEY="sk-your-openai-key-here"

# Hoặc permanent (System Environment Variables)
# Win + R -> sysdm.cpl -> Advanced -> Environment Variables
# Thêm OPENAI_API_KEY = sk-your-key
```

### Bước 3: Khởi động server

**Option A: Tự động (Khuyến nghị)**
```powershell
.\start_system.bat
```
- Tự động mở API server + NGrok
- Tự động mở Swagger docs tại http://localhost:5000/docs

**Option B: Thủ công**
```powershell
# Terminal 1: Start API server
python api_server.py

# Terminal 2: Start NGrok
ngrok http 5000
```

### Bước 4: Test API

**Option 1: Swagger UI (Interactive)**
- Mở browser: http://localhost:5000/docs
- Test từng endpoint trực tiếp trên web

**Option 2: Test Script**
```powershell
# Test với file WAV có sẵn
python test_api_local.py your_audio.wav

# Hoặc test với WAV tự động tạo
python test_api_local.py
```

**Option 3: cURL**
```powershell
# Health check
curl http://localhost:5000/health

# Process audio
curl -X POST -F "audio=@test.wav" http://localhost:5000/process_audio -o response.wav
```

---

## 📚 KIẾN TRÚC CODE

### Class Structure

```
core_openai.py (OOP Architecture)
├── MeiRoboConfig        # Configuration dataclass
├── RAGService          # Vector store & retrieval
├── STTService          # Speech-to-Text
├── LLMService          # OpenAI GPT + RAG
├── TTSService          # Text-to-Speech
└── MeiRoboPipeline     # Full pipeline orchestration

api_server.py (FastAPI)
├── Uses MeiRoboPipeline
├── RESTful endpoints
└── Swagger docs auto-generated
```

### Key Differences from Old Version

**✅ Advantages:**
1. **Class-based:** Clean OOP structure, easy to maintain
2. **FastAPI:** 
   - Automatic API docs (Swagger)
   - Type hints + validation
   - Better performance (async support)
   - Modern Python ecosystem
3. **Separation of concerns:** Each service is independent
4. **Reusable:** Can import classes for different use cases
5. **Short API code:** `api_server.py` is only ~150 lines

---

## 🔧 CONFIGURATION

### Thay đổi cấu hình

Edit trong [api_server.py](api_server.py) trước khi start:

```python
# Tạo custom config
from core_openai import MeiRoboConfig, MeiRoboPipeline

config = MeiRoboConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    llm_model="gpt-4o-mini",      # Hoặc "gpt-4o"
    tts_voice="nova",              # nova, alloy, echo, fable, onyx, shimmer
    tts_speed=1.2,                 # Tăng tốc độ nói
    rag_k=1,                       # Giảm docs retrieve (nhanh hơn)
    max_tokens=300,                # Giảm độ dài response
    temperature=0.7                # Tăng creativity
)

pipeline = MeiRoboPipeline(config)
```

---

## 🌐 API ENDPOINTS

### GET /
Root endpoint, trả về thông tin service

### GET /health
Health check, kiểm tra server đang chạy

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "llm_model": "gpt-4o-mini",
  "tts_voice": "nova"
}
```

### POST /process_audio
Main endpoint: WAV in → WAV out

**Request:**
- `audio`: WAV file (16kHz mono 16-bit recommended)

**Response:**
- WAV file with speech
- Headers:
  - `X-Processing-Time`: Total time (seconds)
  - `X-User-Text`: Transcribed text
  - `X-Reply-Text`: LLM response text

### POST /reset
Reset conversation history

**Response:**
```json
{
  "status": "ok",
  "message": "Conversation history reset"
}
```

### GET /stats
Get system statistics

**Response:**
```json
{
  "config": {
    "device": "cuda",
    "llm_model": "gpt-4o-mini",
    "tts_model": "tts-1",
    "tts_voice": "nova",
    "rag_k": 2
  },
  "conversation_length": 5
}
```

---

## 📊 PERFORMANCE TIPS

### 1. Giảm Latency

```python
# Trong api_server.py hoặc core_openai.py
config = MeiRoboConfig(
    rag_k=1,           # Từ 2 → 1
    max_tokens=300,    # Từ 500 → 300
    tts_speed=1.3      # Từ 1.0 → 1.3
)
```

### 2. GPU Acceleration

- Đảm bảo PyTorch detect được CUDA
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Nếu False → cài `torch` với CUDA support

### 3. Caching

Implement caching cho repeated queries (tùy chọn):

```python
# Trong LLMService
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_chat(self, user_text: str):
    # ...
```

---

## 🐛 TROUBLESHOOTING

### Lỗi: "OPENAI_API_KEY not set"
```powershell
$env:OPENAI_API_KEY="sk-your-key"
```

### Lỗi: "Cannot load FAISS index"
- Check thư mục `vector_stores/faiss_index/` tồn tại
- Phải có files: `index.faiss`, `index.pkl`

### Lỗi: Port 5000 đã được dùng
```powershell
# Tìm process dùng port 5000
netstat -ano | findstr :5000

# Kill process (thay PID)
taskkill /PID <PID> /F

# Hoặc đổi port trong api_server.py
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Server chậm lần đầu
- Lần đầu load RAG embeddings model (~400MB)
- Lần sau nhanh hơn (cached)

### FastAPI docs không mở
- Check server đang chạy: http://localhost:5000/health
- Truy cập thủ công: http://localhost:5000/docs

---

## 🔄 MIGRATION FROM OLD CODE

### Nếu bạn đã có code cũ (Flask version)

**core_openai.py:**
- ✅ Đã refactor thành classes
- ✅ Tất cả functions cũ vẫn hoạt động qua pipeline
- ✅ Có thể dùng `main()` để test local

**api_server.py:**
- ✅ Đổi từ Flask → FastAPI
- ✅ Code ngắn hơn (~150 vs ~350 lines)
- ✅ Tất cả endpoints tương thích
- ⚠️ Response format giống nhau, nhưng FastAPI có thêm auto docs

**robot_client.cpp:**
- ✅ Không thay đổi gì
- ✅ Vẫn gọi `/process_audio` như cũ

---

## 📝 NEXT STEPS

1. **Test local:** `python test_api_local.py`
2. **Setup NGrok:** Get public URL
3. **Update robot code:** Paste NGrok URL vào `robot_client.cpp`
4. **Deploy to Jetson:** Compile và chạy trên robot

Xem thêm: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) cho hướng dẫn đầy đủ
