# Intent ID Protocol - Optimization Guide

## 📊 Tổng Quan

Thay vì gửi file WAV qua network (có thể 100KB-1MB), server sẽ gửi **1 byte** (intent_id) về client. Client sẽ play file WAV tương ứng được lưu locally.

### ⚡ Lợi Ích
- **Giảm latency**: ~1-2s (không cần TTS + chuyển đổi format)
- **Giảm bandwidth**: 1 byte thay vì 100KB+ file WAV
- **Offline**: Client có thể play file WAV ngay lập tức, không phụ thuộc vào mạng

---

## 🔢 Intent ID Mapping

| Intent ID | Tên Intent | File WAV | Nội Dung |
|-----------|-----------|----------|---------|
| **0** | `robot_intro` | `Meirobot.wav` | "Tôi là MeiRobo, robot nhân hình của Meiko Automation" |
| **1** | `company_intro` | `MeikoIntro.wav` | "Meiko Automation chuyên về tự động hóa công nghiệp" |
| **2** | `product_intro` | `ProductIntroduce.wav` | "Sản phẩm của Meiko Automation bao gồm..." |
| **3** | `new_year_greeting` | `CMNM.wav` | "Chúc mừng năm mới!" |

---

## 📡 Server Response Format

### Cách Hoạt Động

```
┌─────────────────────┐
│  Robot (C++)        │
│  Gửi: WAV bytes     │
└──────────┬──────────┘
           │ HTTP POST /process_audio
           ▼
┌──────────────────────────────────────────┐
│  Server (core_openai.py)                 │
│  1. STT: WAV → user_text                 │
│  2. Intent Detection: user_text → intent_id
│  3. Trả về: intent_id (1 byte)           │
└──────────┬──────────────────────────────┘
           │ Response: {"response": b'\x00', "timings": {...}}
           ▼
┌─────────────────────┐
│  Robot (C++)        │
│  1. Nhận intent_id  │
│  2. Play file local │
│  3. Example:        │
│     - intent_id=0   │
│     → Play Meirobot.wav
│     - intent_id=1   │
│     → Play MeikoIntro.wav
└─────────────────────┘
```

### Response Format

```json
{
  "response": "<1-byte intent_id hoặc WAV data>",
  "timings": {
    "stt": 0.45,
    "rag": 0.0,
    "llm": 0.0,
    "tts": 0.0,
    "total": 0.45,
    "user_text": "Bạn là ai?",
    "reply": "Tôi là MeiRobo...",
    "intent_id": 0,
    "intent_wav_file": "Meirobot.wav",
    "canned_response": true
  }
}
```

---

## 💾 Client Setup

### Bước 1: Lưu Files WAV Locally

Client cần lưu 4 file WAV này:

```
/robot/audio/canned_responses/
├── Meirobot.wav          (Intent ID 0)
├── MeikoIntro.wav        (Intent ID 1)
├── ProductIntroduce.wav  (Intent ID 2)
└── CMNM.wav              (Intent ID 3)
```

### Bước 2: Parse Response

```python
# Pseudocode for C++ client
def process_server_response(response_bytes, timings):
    # Check if this is an intent_id (1 byte) or WAV
    if len(response_bytes) == 1 and timings.get('canned_response'):
        # This is intent_id
        intent_id = response_bytes[0]
        
        # Map intent_id to filename
        files = {
            0: "Meirobot.wav",
            1: "MeikoIntro.wav",
            2: "ProductIntroduce.wav",
            3: "CMNM.wav"
        }
        
        wav_file = files[intent_id]
        
        # Play the WAV file locally
        play_audio_file(f"audio/canned_responses/{wav_file}")
        
        print(f"⚡ Intent response: {intent_id} ({timings['total']:.2f}s)")
    else:
        # This is normal WAV data from LLM+TTS pipeline
        play_audio_bytes(response_bytes)
        print(f"LLM response: {timings['total']:.2f}s")
```

---

## 🎯 Intent Detection Patterns

### Intent 0: Robot Introduction
**Trigger Patterns:**
- Exact: "giới thiệu bản thân", "bạn là ai", "tên bạn là gì"
- Contains: "bạn.*là.*ai", "tên.*của.*bạn"

### Intent 1: Company Introduction
**Trigger Patterns:**
- Exact: "giới thiệu công ty", "công ty làm gì"
- Contains: "meiko.*làm.*gì", "về.*công ty"

### Intent 2: Product Introduction
**Trigger Patterns:**
- Exact: "giới thiệu sản phẩm", "các sản phẩm"
- Contains: "sản phẩm.*meiko", "công ty.*có.*gì"

### Intent 3: New Year Greeting
**Trigger Patterns:**
- Exact: "chúc mừng năm mới", "chúc tết"
- Contains: "chúc.*mừng.*năm", "chúc.*tết"

---

## 📊 Latency Comparison

### Trước (Old Way)
```
STT:           0.5s
LLM:           1.2s
TTS:           1.0s (synthesize + format conversion)
File I/O:      0.1s
Total:         2.8s
Bandwidth:     100KB+ (WAV file)
```

### Sau (New Way) - Intent Match
```
STT:           0.5s
Intent Det:    0.01s (regex matching)
Total:         0.51s ⚡⚡⚡ (5.5x faster!)
Bandwidth:     1 byte (intent_id)
```

### Sau (New Way) - No Intent Match
```
STT:           0.5s
RAG:           0.2s
LLM:           1.2s
TTS:           1.0s
Total:         2.9s (same as before)
Bandwidth:     100KB+ (WAV file)
```

---

## 🔄 Integration Checklist

### Server Side (Already Done ✅)
- [x] Add `intent_id` to IntentDetector
- [x] Return intent_id as 1-byte instead of WAV
- [x] Include metadata in response: `intent_id`, `intent_wav_file`, `canned_response`
- [x] Skip TTS for intent matches

### Client Side (TODO)
- [ ] Pre-cache 4 WAV files locally
- [ ] Parse response to detect intent_id
- [ ] Map intent_id → filename
- [ ] Play WAV file from local cache
- [ ] Fallback to playing response_bytes if not intent_id

---

## 📝 Example Response

### Intent Match
```
Request: WAV bytes (robot asking "Bạn là ai?")

Response HTTP:
200 OK
{
  "response": "\x00",  // 1-byte: intent_id = 0
  "timings": {
    "stt": 0.45,
    "total": 0.45,
    "user_text": "Bạn là ai?",
    "reply": "Tôi là MeiRobo, robot nhân hình của Meiko Automation",
    "intent_id": 0,
    "intent_wav_file": "Meirobot.wav",
    "canned_response": true
  }
}

Client Action:
1. Detect: response is 1 byte + canned_response=true
2. Extract: intent_id = 0
3. Play: audio/canned_responses/Meirobot.wav
4. Total latency: 0.45s
```

### No Intent Match
```
Request: WAV bytes (robot asking "Meiko Automation có những công nghệ gì?")

Response HTTP:
200 OK
{
  "response": "<WAV bytes - normal audio data>",
  "timings": {
    "stt": 0.45,
    "rag": 0.15,
    "llm": 1.2,
    "tts": 0.8,
    "total": 2.6,
    "user_text": "Meiko Automation có những công nghệ gì?",
    "reply": "Meiko Automation chuyên về tự động hóa...",
    "canned_response": false
  }
}

Client Action:
1. Detect: response is large + canned_response=false
2. Play: response as audio bytes
3. Total latency: 2.6s
```

---

## 🚀 Future Enhancements

1. **Add more intents** (scale from 4 to 10, 20, ...)
   - Intent ID can go up to 255 (single byte)
   - Just add to `intent_patterns` in IntentDetector

2. **Multi-language intent detection**
   - Add English, Chinese patterns alongside Vietnamese

3. **Dynamic WAV file updates**
   - Send new audio files to client without rebuilding
   - Use intent_id as stable identifier

4. **Intent confidence scoring**
   - Return intent_id only if confidence > 0.9
   - Otherwise fallback to LLM

---

## ⚠️ Notes

- **Intent IDs must be 0-3** (4 intents = 4 bytes = 0-3)
- **Client MUST have all 4 WAV files pre-cached**
- If file missing on client, it should fallback gracefully
- Server logs will show: "⚡ INTENT MATCH (ID: X)"
