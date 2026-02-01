# core_openai.py
# =====================================================
# AI Server Core - WAV to WAV Pipeline for Robot
# Optimized for: C++ Robot → AI Server → C++ Robot
# =====================================================

import os
import io
import time
import wave
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import speech_recognition as sr
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from scipy import signal

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =====================================================
# CONFIGURATION
# =====================================================

@dataclass
class MeiRoboConfig:
    """Configuration for MeiRobo AI Server"""
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    tts_model: str = "gpt-4o-mini-tts"  # gpt-4o-mini-tts for low latency
    tts_voice: str = "nova"  # alloy, echo, fable, onyx, nova, shimmer
    tts_speed: float = 1.2
    tts_output_sample_rate: int = 16000  # Target sample rate for robot (convert after API call)
    tts_output_channels: int = 1  # Target channels for robot (convert after API call)
    stt_language: str = "vi-VN"
    system_prompt: str = """Bạn tên là MeiRobo, humanoid và là nhân viên của công ty Meiko Automation. Giám đốc hiện tại là sếp Nguyễn Văn Thuận. Và tôi có thể biểu diễn nhảy và múa võ, luôn sẵn sàng biểu diễn cho mọi người nhân dịp cuối năm"
Cách trả lời:
- Trả lời ngắn gọn, tự nhiên như hai người nói chuyện bình thường.
- Khi nói về Meiko Automation, LUÔN dùng ngôi thứ nhất.
- Không hỏi lại người dùng ở cuối câu.
- Nếu hỏi về công ty, dùng CONTEXT để trả lời ngắn gọn, đời thường.
Quy tắc:
- Mỗi câu trả lời tối đa 1–2 câu.
"""

    vector_db_path: str = "vector_stores/faiss_index"
    rag_k: int = 3
    max_tokens: int = 200
    temperature: float = 0.2
    timeout: float = 30.0
    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# RAG SERVICE
# =====================================================

class RAGService:
    """Retrieval-Augmented Generation Service"""
    
    def __init__(self, config: MeiRoboConfig):
        self.config = config
        print(f"📚 Loading RAG on device: {config.device}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": config.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        self.vectorstore = FAISS.load_local(
            config.vector_db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.rag_k}
        )
        
        # Cache for last warmup embedding (for parallel processing)
        self._warmup_embedding = None
        
        print("✅ RAG initialized")
    
    def warm_embeddings(self, dummy_text: str = "warm up query") -> float:
        """
        Pre-warm embedding model by encoding a dummy text
        This ensures the model is ready when we need to retrieve context
        
        Returns: warmup_time in seconds
        """
        start = time.time()
        try:
            # This triggers the embedding model to load/warm up
            self._warmup_embedding = self.embeddings.embed_query(dummy_text)
        except Exception as e:
            print(f"⚠️ Embedding warmup warning: {e}")
        return time.time() - start
    
    def retrieve_context(self, query: str, max_chars: int = 200) -> Tuple[str, float]:
        """
        Retrieve relevant context for a query
        Returns: (context, retrieval_time)
        """
        start = time.time()
        docs = self.retriever.invoke(query)
        context = "\n".join(d.page_content[:max_chars] for d in docs)
        elapsed = time.time() - start
        return context, elapsed


# =====================================================
# STT SERVICE
# =====================================================

class STTService:
    """Speech-to-Text Service with Google + Whisper fallback"""
    
    def __init__(self, config: MeiRoboConfig):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.openai_client = OpenAI(api_key=config.openai_api_key, timeout=config.timeout)
        self.use_whisper_fallback = False  # Enable fallback
        print("🎤 STT Service initialized (Google + Whisper fallback)")
    
    def preprocess_audio(self, wav_bytes: bytes, target_sample_rate: int = 16000) -> bytes:
        """
        Preprocess audio: fix speed issues, resample if needed
        Args:
            wav_bytes: Input WAV file bytes
            target_sample_rate: Target sample rate (default 16kHz for STT)
        Returns:
            Preprocessed WAV bytes
        """
        try:
            # Read WAV file properties
            with io.BytesIO(wav_bytes) as f:
                with wave.open(f, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            if sampwidth == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sampwidth == 4:  # 32-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
            
            # Handle stereo -> mono
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(audio_array.dtype)
            
            # Resample if needed (fix speed issues)
            if framerate != target_sample_rate:
                print(f"🔧 Resampling: {framerate}Hz -> {target_sample_rate}Hz")
                num_samples = int(len(audio_array) * target_sample_rate / framerate)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
            
            # Create new WAV file with correct parameters
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_out:
                wav_out.setnchannels(1)  # Mono
                wav_out.setsampwidth(2)  # 16-bit
                wav_out.setframerate(target_sample_rate)
                wav_out.writeframes(audio_array.astype(np.int16).tobytes())
            
            return output.getvalue()
            
        except Exception as e:
            print(f"⚠️ Preprocessing warning: {e}, using original audio")
            return wav_bytes
    
    def transcribe_with_whisper(self, wav_bytes: bytes) -> Tuple[str, float]:
        """
        Fallback: Transcribe using OpenAI Whisper API
        Args:
            wav_bytes: WAV file bytes
        Returns: (text, time_taken)
        """
        start = time.time()
        try:
            # OpenAI Whisper API requires file-like object with name
            audio_file = io.BytesIO(wav_bytes)
            audio_file.name = "audio.wav"
            
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="vi"  # Vietnamese
            )
            
            elapsed = time.time() - start
            return transcript.text.strip(), elapsed
            
        except Exception as e:
            elapsed = time.time() - start
            print(f"❌ Whisper Error: {e}")
            return "", elapsed
    
    def transcribe_from_wav_bytes(self, wav_bytes: bytes, preprocess: bool = False) -> Tuple[str, Dict[str, float]]:
        """
        Transcribe audio from WAV bytes (from robot)
        Uses Google Speech Recognition with Whisper fallback
        
        Args:
            wav_bytes: Input WAV file bytes
            preprocess: Whether to preprocess audio (fix speed, resample) - Default: False for speed
        Returns: (text, timing_dict)
        """
        timings = {}
        start = time.time()
        
        try:
            # Preprocess audio only if needed (optional, disabled by default for speed)
            if preprocess:
                t_pre = time.time()
                wav_bytes = self.preprocess_audio(wav_bytes)
                timings['preprocess'] = time.time() - t_pre
            else:
                timings['preprocess'] = 0.0  # Skipped
            
            # Try Google Speech Recognition first (fast, free)
            text = ""
            try:
                # Convert WAV bytes to AudioData
                with io.BytesIO(wav_bytes) as audio_file:
                    with sr.AudioFile(audio_file) as source:
                        audio = self.recognizer.record(source)
                
                # Google Speech Recognition
                t1 = time.time()
                text = self.recognizer.recognize_google(
                    audio, 
                    language=self.config.stt_language
                ).strip()
                timings['google'] = time.time() - t1
                timings['method'] = 'google'
                
            except (sr.UnknownValueError, sr.RequestError) as e:
                print(f"⚠️ Google STT failed: {e}")
                text = ""
            
            # Fallback to Whisper if Google failed
            if not text and self.use_whisper_fallback:
                print("🔄 Falling back to Whisper...")
                text, whisper_time = self.transcribe_with_whisper(wav_bytes)
                timings['whisper'] = whisper_time
                timings['method'] = 'whisper'
            
            timings['total'] = time.time() - start
            return text, timings
            
        except Exception as e:
            timings['total'] = time.time() - start
            print(f"❌ STT Error: {e}")
            return "", timings


# =====================================================
# LLM SERVICE
# =====================================================

# =====================================================
# LLM SERVICE (Responses API - Optimized)
# =====================================================

class LLMService:
    """
    LLM Service using OpenAI Responses API
    Optimized for:
    - Low latency
    - Conversation state
    - Robot real-time interaction
    """

    def __init__(self, config: MeiRoboConfig, rag_service: Optional[RAGService] = None):
        self.config = config
        self.rag_service = rag_service
        self.client = OpenAI(
            api_key=config.openai_api_key,
            timeout=config.timeout
        )

        # ⭐ Conversation state handled by OpenAI
        self.last_response_id: Optional[str] = None

        print(f"🧠 LLM Service initialized (Responses API | model={config.llm_model})")

    def chat(
        self,
        user_text: str,
        use_rag: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate response using Responses API
        Returns: (reply, timing_dict)
        """
        timings = {}
        start = time.time()

        try:
            # =================================================
            # 1. RAG retrieval (optional)
            # =================================================
            # Skip RAG for simple queries (greetings, thanks, etc.)
            # This saves ~0.1-0.3s per query
            import re
            
            user_lower = user_text.lower().strip()
            
            # ════════════════════════════════════════════════════════════
            # SIMPLE QUERY PATTERNS - Skip RAG for these
            # ════════════════════════════════════════════════════════════
            
            # 1. EXACT match patterns (câu chính xác)
            exact_patterns = [
                # Greetings - Chào hỏi
                r"^xin chào$", r"^chào$", r"^chào bạn$", r"^chào nhé$",
                r"^hello$", r"^hi$", r"^hey$",
                
                # Thanks - Cảm ơn
                r"^cảm ơn$", r"^cảm ơn bạn$", r"^cảm ơn nhé$", r"^cám ơn$",
                r"^thank(s)?$", r"^thank you$",
                
                # Goodbye - Tạm biệt
                r"^tạm biệt$", r"^bye$", r"^goodbye$", r"^bye bye$",
                r"^hẹn gặp lại$", r"^gặp lại nhé$",
                
                # Identity - Hỏi về robot
                r"^bạn tên (là )?gì\??$", r"^tên bạn là gì\??$", 
                r"^bạn là ai\??$", r"^bạn là gì\??$",
                r"^mày là ai\??$", r"^bạn là robot à\??$",
                
                # Simple responses
                r"^ok$", r"^okay$", r"^được$", r"^vâng$", r"^dạ$",
                r"^ừ$", r"^ờ$", r"^à$", r"^uh huh$",
                
                # Affirmations
                r"^có$", r"^không$", r"^yes$", r"^no$",
                r"^đúng$", r"^sai$", r"^đúng rồi$",
            ]
            
            # 2. START WITH patterns (câu bắt đầu bằng)
            start_patterns = [
                # Greetings that start sentences
                r"^xin chào\b",      # "Xin chào, tôi là..."
                r"^chào bạn\b",      # "Chào bạn, tôi muốn..."
                r"^hello\b",         # "Hello, I am..."
                r"^hi\b",            # "Hi there..."
                
                # Thanks that start sentences
                r"^cảm ơn\b",        # "Cảm ơn bạn đã giúp..."
                r"^cám ơn\b",        # "Cám ơn nhiều..."
                
                # Goodbye that start sentences
                r"^tạm biệt\b",      # "Tạm biệt nhé..."
                r"^bye\b",           # "Bye, see you..."
            ]
            
            # 3. CONTAINS patterns (câu chứa từ khóa - nhưng ngắn)
            # Chỉ áp dụng cho câu ngắn (< 30 ký tự) để tránh false positive
            short_query_keywords = [
                "khỏe không", "có khỏe không", "bạn khỏe không",
                "thế nào", "sao rồi", "dạo này sao",
            ]
            
            # ════════════════════════════════════════════════════════════
            # CHECK PATTERNS
            # ════════════════════════════════════════════════════════════
            
            is_exact_simple = any(re.match(p, user_lower) for p in exact_patterns)
            is_greeting_start = any(re.match(p, user_lower) for p in start_patterns)
            is_short_simple = (
                len(user_lower) < 30 and 
                any(kw in user_lower for kw in short_query_keywords)
            )
            
            is_simple_query = is_exact_simple or is_greeting_start or is_short_simple
            
            # ════════════════════════════════════════════════════════════
            # RAG RETRIEVAL
            # ════════════════════════════════════════════════════════════
            
            context = ""
            if use_rag and self.rag_service and not is_simple_query:
                context, rag_time = self.rag_service.retrieve_context(user_text)
                timings["rag"] = rag_time
                timings["rag_context"] = context  # Store context for logging
                
                # 📝 In ra thông tin RAG context đã truy vấn
                print(f"\n{'─'*60}")
                print(f"📚 RAG CONTEXT RETRIEVED ({rag_time:.2f}s)")
                print(f"{'─'*60}")
                print(f"🔍 Query: {user_text}")
                print(f"📄 Context: {context[:300]}{'...' if len(context) > 300 else ''}")
                print(f"{'─'*60}")
                
            elif is_simple_query:
                timings["rag"] = 0.0  # Skipped for simple query
                timings["rag_context"] = ""  # No context
                
                # Log skip reason
                if is_exact_simple:
                    print(f"📝 RAG skipped (exact match: '{user_text}')")
                elif is_greeting_start:
                    print(f"📝 RAG skipped (greeting start: '{user_text[:40]}{'...' if len(user_text) > 40 else ''}')")
                else:
                    print(f"📝 RAG skipped (short simple: '{user_text}')")
            else:
                timings["rag"] = 0.0
                timings["rag_context"] = ""

            # =================================================
            # 2. Build input text
            # =================================================
            if context.strip():
                input_text = f"USER:\n{user_text}\n\nCONTEXT:\n{context}"
            else:
                input_text = user_text

            # =================================================
            # 3. Call OpenAI Responses API
            # =================================================
            t_llm = time.time()

            response = self.client.responses.create(
                model=self.config.llm_model,

                # ⭐ System-level instruction (NOT repeated in history)
                instructions=self.config.system_prompt,

                # ⭐ User input
                input=input_text,

                # ⭐ Keep conversation state on OpenAI side
                previous_response_id=self.last_response_id,

                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )

            timings["llm"] = time.time() - t_llm

            # =================================================
            # 4. Extract text safely
            # =================================================
            reply = response.output_text.strip() if response.output_text else ""

            # =================================================
            # 5. Save response_id for next turn
            # =================================================
            self.last_response_id = response.id

            timings["total"] = time.time() - start

            return reply or "Mình chưa nghe rõ.", timings

        except Exception as e:
            timings["total"] = time.time() - start
            print(f"❌ LLM Error (Responses API): {e}")
            return "Xin lỗi, hệ thống đang gặp sự cố.", timings

    def reset_conversation(self):
        """
        Reset conversation state
        """
        self.last_response_id = None
        print("🔄 Conversation context reset (Responses API)")


# =====================================================
# INTENT DETECTION & CANNED RESPONSES
# =====================================================

class IntentDetector:
    """
    Phát hiện intent (ý định) của câu hỏi và trả về intent_id (số)
    Client sẽ lưu các file WAV, server chỉ gửi intent_id để chỉ file nào play
    ⚡ Giúp giảm latency & bandwidth - không cần gửi file WAV qua network
    """
    
    def __init__(self, canned_responses_dir: str = "audiocases_rep"):
        self.canned_responses_dir = canned_responses_dir
        
        # ════════════════════════════════════════════════════════════
        # INTENT PATTERNS - Định nghĩa các pattern cho từng intent
        # Intent ID mapping: 0=robot_intro, 1=company_intro, 2=product_intro, 3=new_year_greeting
        # ════════════════════════════════════════════════════════════
        
        self.intent_patterns = {
            # Intent 0: Giới thiệu bản thân robot
            "robot_intro": {
                "intent_id": 0,
                "wav_file": "Meirobot.wav",
                "text_response": "Tôi là MeiRobo, robot nhân hình của Meiko Automation",
                "patterns": [
                    # Exact matches
                    r"^giới thiệu bản thân$",
                    r"^bạn là ai$",
                    r"^tên bạn là gì$",
                    r"^bạn tên gì$",
                    r"^cho mình biết về bạn$",
                    
                    # Contains keywords
                    r"giới thiệu.*bản thân",
                    r"giới thiệu.*về bạn",
                    r"bạn.*là.*ai",
                    r"tên.*của.*bạn",
                    r"ai.*là.*bạn",
                ]
            },
            
            # Intent 1: Giới thiệu công ty Meiko Automation
            "company_intro": {
                "intent_id": 1,
                "wav_file": "MeikoIntro.wav",
                "text_response": "Meiko Automation chuyên về tự động hóa công nghiệp",
                "patterns": [
                    # Exact matches
                    r"^giới thiệu công ty$",
                    r"^công ty làm gì$",
                    r"^meiko làm gì$",
                    r"^meiko automation là gì$",
                    
                    # Contains keywords
                    r"giới thiệu.*công ty",
                    r"giới thiệu.*meiko",
                    r"công ty.*meiko.*làm gì",
                    r"meiko.*automation.*làm gì",
                    r"meiko.*chuyên.*gì",
                    r"về.*công ty.*meiko",
                    r"cho.*biết.*về.*meiko",
                ]
            },
            
            # Intent 2: Giới thiệu sản phẩm
            "product_intro": {
                "intent_id": 2,
                "wav_file": "ProductIntroduce.wav",
                "text_response": "Sản phẩm của Meiko Automation bao gồm...",
                "patterns": [
                    # Exact matches
                    r"^giới thiệu sản phẩm$",
                    r"^sản phẩm của công ty$",
                    r"^các sản phẩm$",
                    
                    # Contains keywords
                    r"giới thiệu.*sản phẩm",
                    r"sản phẩm.*của.*meiko",
                    r"sản phẩm.*của.*công ty",
                    r"meiko.*có.*sản phẩm.*gì",
                    r"các.*sản phẩm.*của.*meiko",
                    r"cho.*biết.*sản phẩm",
                ]
            },
            
            # Intent 3: Chúc mừng năm mới
            "new_year_greeting": {
                "intent_id": 3,
                "wav_file": "CMNM.wav",
                "text_response": "Chúc mừng năm mới!",
                "patterns": [
                    # Exact matches
                    r"^chúc mừng năm mới$",
                    r"^chúc tết$",
                    r"^happy new year$",
                    
                    # Contains keywords
                    r"chúc.*mừng.*năm.*mới",
                    r"chúc.*năm.*mới",
                    r"lời.*chúc.*năm.*mới",
                    r"gửi.*lời.*chúc",
                    r"chúc.*tết",
                    r"chúc.*cán bộ.*công nhân viên",
                ]
            },
            # Intent 4: Nhảy, múa võ
            "dance_martial": {
                "intent_id": 4,
                "wav_file": "DanceMartial.wav",
                "text_response": "Tôi có thể biểu diễn nhảy và múa võ bất cứ lúc nào! Bạn muốn xem không?",
                "patterns": [
                    # Exact matches
                    r"^bạn có thể nhảy không$",
                    r"^bạn có thể múa võ không$",
                    r"^bạn múa võ đi$",
                    r"^bạn nhảy đi$",
                    r"^biểu diễn nhảy$",
                    r"^biểu diễn múa võ$",
                    r"^có thể nhảy không$",
                    r"^có thể múa võ không$",
                    # Contains keywords
                    r"biểu diễn.*nhảy",
                    r"biểu diễn.*múa võ",
                    r"nhảy.*được không",
                    r"múa võ.*được không",
                    r"bạn.*nhảy",
                    r"bạn.*múa võ",
                    r"cho.*xem.*nhảy",
                    r"cho.*xem.*múa võ",
                ]
            },
        }
        
        # Build intent ID → data mapping for quick lookup
        self.intent_id_map = {intent_data["intent_id"]: (intent_name, intent_data) 
                              for intent_name, intent_data in self.intent_patterns.items()}
        
        print(f"🎯 Intent Detector initialized with {len(self.intent_patterns)} intents")
        print(f"   Intent ID Mapping:")
        for intent_id, (intent_name, _) in sorted(self.intent_id_map.items()):
            print(f"      {intent_id} = {intent_name}")
    
    def detect_intent(self, user_text: str) -> Optional[Tuple[int, str, str]]:
        """
        Phát hiện intent từ câu hỏi của user
        ⚡ Gửi intent_id (số) thay vì file WAV để giảm bandwidth
        
        Args:
            user_text: Câu hỏi của user (đã lowercase)
            
        Returns:
            (intent_id, wav_filename, text_response) nếu match
            None nếu không match
        """
        import re
        
        # Normalize text
        text_lower = user_text.lower().strip()
        
        # Remove punctuation
        text_lower = re.sub(r'[?!.,;:]', '', text_lower)
        
        # Try to match each intent
        for intent_name, intent_data in self.intent_patterns.items():
            patterns = intent_data["patterns"]
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Match found!
                    intent_id = intent_data["intent_id"]
                    wav_filename = intent_data["wav_file"]  # Just filename, not full path
                    text_response = intent_data["text_response"]
                    
                    print(f"🎯 Intent detected: {intent_name} (ID: {intent_id})")
                    print(f"   Pattern matched: {pattern}")
                    print(f"   WAV file: {wav_filename}")
                    
                    return intent_id, wav_filename, text_response
        
        # No match
        return None


# =====================================================
# TTS SERVICE
# =====================================================

class TTSService:
    """Text-to-Speech Service using OpenAI"""
    
    def __init__(self, config: MeiRoboConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key, timeout=config.timeout)
        print(f"🔊 TTS Service initialized")
        print(f"   Model: {config.tts_model}")
        print(f"   Voice: {config.tts_voice}")
        print(f"   Output: {config.tts_output_sample_rate}Hz, {config.tts_output_channels}-channel")
    
    def convert_to_target_format(self, wav_bytes: bytes) -> bytes:
        """
        Convert WAV to target format (16kHz mono for robot)
        
        Args:
            wav_bytes: Input WAV bytes from OpenAI (24kHz by default)
            
        Returns:
            Converted WAV bytes (16kHz mono)
        """
        try:
            # Read WAV properties
            with io.BytesIO(wav_bytes) as f:
                with wave.open(f, 'rb') as wav_file:
                    n_channels = wav_file.getnchannels()
                    sampwidth = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_data = wav_file.readframes(n_frames)
            
            # Already in target format? Return as-is
            if (framerate == self.config.tts_output_sample_rate and 
                n_channels == self.config.tts_output_channels):
                return wav_bytes
            
            # Convert to numpy array
            if sampwidth == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            elif sampwidth == 4:  # 32-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int32)
                audio_array = (audio_array / 65536).astype(np.int16)
            else:
                audio_array = np.frombuffer(audio_data, dtype=np.uint8)
            
            # Stereo to mono
            if n_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
            
            # Resample if needed
            if framerate != self.config.tts_output_sample_rate:
                num_samples = int(len(audio_array) * self.config.tts_output_sample_rate / framerate)
                audio_array = signal.resample(audio_array, num_samples).astype(np.int16)
            
            # Create output WAV
            output = io.BytesIO()
            with wave.open(output, 'wb') as wav_out:
                wav_out.setnchannels(self.config.tts_output_channels)
                wav_out.setsampwidth(2)  # 16-bit
                wav_out.setframerate(self.config.tts_output_sample_rate)
                wav_out.writeframes(audio_array.tobytes())
            
            return output.getvalue()
            
        except Exception as e:
            print(f"⚠️ Convert warning: {e}, returning original")
            return wav_bytes
    
    def synthesize_to_wav_bytes(self, text: str) -> Tuple[bytes, Dict[str, float]]:
        """
        Convert text to speech WAV bytes (for robot)
        Output: 16kHz mono WAV
        
        Returns: (wav_bytes, timing_dict)
        """
        timings = {}
        start = time.time()
        
        try:
            # OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                input=text,
                speed=self.config.tts_speed,
                response_format="wav"
            )
            
            raw_audio = response.content
            timings['tts_api'] = time.time() - start
            
            # Convert to target format (16kHz mono)
            t_convert = time.time()
            wav_bytes = self.convert_to_target_format(raw_audio)
            timings['convert'] = time.time() - t_convert
            timings['total'] = time.time() - start
            
            return wav_bytes, timings
            
        except Exception as e:
            timings['total'] = time.time() - start
            print(f"❌ TTS Error: {e}")
            return b'', timings
    
    def synthesize_streaming(self, text: str, convert_format: bool = True):
        """
        Stream audio chunks from OpenAI TTS API
        
        ⚡ TRUE STREAMING - Follows OpenAI documentation:
        https://platform.openai.com/docs/guides/text-to-speech#streaming-realtime-audio
        
        "The Speech API provides support for realtime audio streaming using 
        chunk transfer encoding. This means the audio can be played BEFORE 
        the full file is generated and made accessible."
        
        Args:
            text: Text to convert to speech
            convert_format: If True, convert to 16kHz mono (slower, not true streaming)
                          If False, return raw OpenAI format (faster, true streaming)
            
        Yields:
            Audio chunks (bytes)
        """
        try:
            if convert_format:
                # ════════════════════════════════════════════════════════════
                # MODE 1: BUFFERED STREAMING (convert to 16kHz mono)
                # Slower but compatible with robot's 16kHz requirement
                # ════════════════════════════════════════════════════════════
                buffer = io.BytesIO()
                
                with self.client.audio.speech.with_streaming_response.create(
                    model=self.config.tts_model,
                    voice=self.config.tts_voice,
                    input=text,
                    speed=self.config.tts_speed,
                    response_format="wav"
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if chunk:
                            buffer.write(chunk)
                
                # Convert complete WAV to 16kHz mono
                raw_wav = buffer.getvalue()
                converted_wav = self.convert_to_target_format(raw_wav)
                
                # Yield converted WAV in chunks
                for i in range(0, len(converted_wav), 8192):
                    yield converted_wav[i:i+8192]
            else:
                # ════════════════════════════════════════════════════════════
                # MODE 2: TRUE STREAMING (raw OpenAI format - 24kHz)
                # ⚡ Fastest - yield chunks immediately as they arrive!
                # Robot must handle 24kHz WAV format
                # ════════════════════════════════════════════════════════════
                with self.client.audio.speech.with_streaming_response.create(
                    model=self.config.tts_model,
                    voice=self.config.tts_voice,
                    input=text,
                    speed=self.config.tts_speed,
                    response_format="wav"  # WAV for fastest response
                ) as response:
                    # ⚡ TRUE STREAMING: Yield chunks immediately!
                    for chunk in response.iter_bytes(chunk_size=4096):
                        if chunk:
                            yield chunk
                
        except Exception as e:
            print(f"❌ TTS Streaming Error: {e}")
            yield b''  # Return empty on error
    
    def synthesize_streaming_pcm(self, text: str):
        """
        ⚡ FASTEST TRUE STREAMING with PCM format
        
        PCM = raw audio samples without header
        - 24kHz sample rate
        - 16-bit signed
        - Little-endian
        - Mono
        
        Use this for lowest latency. Client must handle raw PCM.
        
        Yields:
            Raw PCM chunks (bytes) - 24kHz 16-bit mono
        """
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model=self.config.tts_model,
                voice=self.config.tts_voice,
                input=text,
                speed=self.config.tts_speed,
                response_format="pcm"  # ⚡ PCM for FASTEST response
            ) as response:
                # ⚡ TRUE STREAMING: Yield chunks immediately!
                for chunk in response.iter_bytes(chunk_size=4096):
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            print(f"❌ TTS PCM Streaming Error: {e}")
            yield b''


# =====================================================
# MAIN PIPELINE
# =====================================================

class MeiRoboPipeline:
    """Complete STT -> LLM+RAG -> TTS Pipeline for AI Server"""
    
    def __init__(self, config: MeiRoboConfig):
        self.config = config
        print("\n" + "="*60)
        print("🤖 Initializing MeiRobo Pipeline")
        print("="*60)
        
        # Initialize services
        self.rag_service = RAGService(config)
        self.stt_service = STTService(config)
        self.llm_service = LLMService(config, self.rag_service)
        self.tts_service = TTSService(config)
        
        # ⚡ NEW: Initialize Intent Detector for canned responses
        self.intent_detector = IntentDetector(canned_responses_dir="audiocases_rep")
        
        print("="*60)
        print("✅ MeiRobo Pipeline Ready")
        print("="*60)
    
    def process_audio_bytes(self, wav_bytes: bytes) -> Tuple[bytes, Dict]:
        """
        Main processing pipeline: WAV in → WAV out
        
        Flow:
        1. Receive WAV from robot (C++)
        2. ⚡ PARALLEL: STT + RAG warmup (embedding pre-warm)
        3. ⚡ CHECK INTENT: If match -> return canned response (FAST!)
        4. LLM+RAG: Generate response (if no intent match)
        5. TTS: Convert response to WAV
        6. Return WAV to robot (C++)
        
        Args:
            wav_bytes: Input WAV file bytes from robot
            
        Returns:
            (response_wav_bytes, timing_dict)
        """
        start = time.time()
        timings = {}
        
        # ⚡ PARALLEL: STT + RAG warmup
        # Trong khi STT đang transcribe, pre-warm embedding model
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            stt_future = executor.submit(
                self.stt_service.transcribe_from_wav_bytes, wav_bytes
            )
            warmup_future = executor.submit(
                self.rag_service.warm_embeddings, "warmup query"
            )
            
            # Wait for STT result
            user_text, stt_timings = stt_future.result()
            warmup_time = warmup_future.result()
            
            timings['stt'] = stt_timings
            timings['rag_warmup'] = warmup_time
            print(f"⚡ Parallel: STT {stt_timings.get('total', 0):.2f}s | RAG warmup {warmup_time:.2f}s")
        
        # Handle empty transcription
        if not user_text:
            error_text = "Mình không nghe rõ, bạn nói lại được không?"
            response_wav, tts_timings = self.tts_service.synthesize_to_wav_bytes(error_text)
            timings['tts'] = tts_timings
            timings['total'] = time.time() - start
            timings['user_text'] = ""
            timings['reply'] = error_text
            return response_wav, timings
        
        # ⚡ NEW: Check for canned responses (intent detection)
        intent_result = self.intent_detector.detect_intent(user_text)
        
        if intent_result is not None:
            # Intent matched! Return intent_id instead of WAV
            intent_id, wav_filename, text_response = intent_result
            
            # ⚡ FAST PATH: Just return intent_id + metadata
            # Client will play the pre-cached WAV file locally
            timings['intent_id'] = intent_id
            timings['intent_wav_file'] = wav_filename
            timings['canned_response'] = True
            timings['llm'] = {'total': 0.0}  # Skipped
            timings['tts'] = {'total': 0.0}  # Skipped
            timings['total'] = time.time() - start
            timings['user_text'] = user_text
            timings['reply'] = text_response
            
            print(f"⚡ INTENT MATCH (ID: {intent_id}): {wav_filename} - {text_response[:30]}...")
            print(f"   Total time: {timings['total']:.2f}s (NO LLM/TTS overhead!)")
            
            # Return special response: intent_id as bytes (1 byte is enough: 0-3)
            intent_code = bytes([intent_id])  # Convert intent_id (0-3) to single byte
            return intent_code, timings
        
        # 2. LLM with RAG - Generate response (fallback if no intent match)
        reply, llm_timings = self.llm_service.chat(user_text)
        timings['llm'] = llm_timings
        timings['canned_response'] = False
        
        # 3. TTS - Text to Speech
        response_wav, tts_timings = self.tts_service.synthesize_to_wav_bytes(reply)
        timings['tts'] = tts_timings
        
        # Complete timing info
        timings['total'] = time.time() - start
        timings['user_text'] = user_text
        timings['reply'] = reply
        
        return response_wav, timings
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.llm_service.reset_conversation()
    
    def process_audio_streaming(self, wav_bytes: bytes):
        """
        Streaming version: STT -> LLM+RAG -> TTS (stream)
        
        ⚡ FASTER perceived latency - yields audio chunks immediately!
        
        Flow:
        1. ⚡ PARALLEL: STT + RAG warmup (embedding pre-warm)
        2. LLM+RAG: Generate response
        3. TTS: Stream audio chunks (start playing immediately)
        
        Args:
            wav_bytes: Input WAV file bytes from robot
            
        Yields:
            (chunk, metadata) tuples:
            - First yield: (b'', {"user_text": ..., "reply": ...})
            - Following yields: (audio_chunk, {})
            - Last yield: (b'', {"timings": {...}})
        """
        start = time.time()
        timings = {}
        
        # ⚡ PARALLEL: STT + RAG warmup
        # Trong khi STT đang transcribe, pre-warm embedding model
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            stt_future = executor.submit(
                self.stt_service.transcribe_from_wav_bytes, wav_bytes
            )
            warmup_future = executor.submit(
                self.rag_service.warm_embeddings, "warmup query"
            )
            
            # Wait for STT result
            user_text, stt_timings = stt_future.result()
            warmup_time = warmup_future.result()
            
            timings['stt'] = stt_timings
            timings['rag_warmup'] = warmup_time
            print(f"⚡ Parallel: STT {stt_timings.get('total', 0):.2f}s | RAG warmup {warmup_time:.2f}s")
        
        # Handle empty transcription
        if not user_text:
            error_text = "Mình không nghe rõ, bạn nói lại được không?"
            yield b'', {"user_text": "", "reply": error_text, "error": True}
            
            for chunk in self.tts_service.synthesize_streaming(error_text):
                yield chunk, {}
            
            timings['total'] = time.time() - start
            yield b'', {"timings": timings}
            return
        
        # 2. LLM with RAG - Generate response
        reply, llm_timings = self.llm_service.chat(user_text)
        timings['llm'] = llm_timings
        
        # Send metadata with timing (so robot can display text + know STT/LLM time)
        yield b'', {
            "user_text": user_text, 
            "reply": reply,
            "stt_time": stt_timings.get('total', 0),
            "llm_time": llm_timings.get('total', 0)
        }
        
        # 3. TTS - Stream audio chunks
        # Using TTSService.synthesize_streaming() method
        # convert_format=True để đảm bảo output 16kHz mono cho robot
        tts_start = time.time()
        for chunk in self.tts_service.synthesize_streaming(reply, convert_format=True):
            yield chunk, {}
        
        timings['tts'] = {'total': time.time() - tts_start}
        timings['total'] = time.time() - start
        
        # Send final timings
        yield b'', {"timings": timings}
    
    def process_audio_streaming_true(self, wav_bytes: bytes):
        """
        ⚡ TRUE STREAMING version - NO format conversion!
        
        Output: WAV 24kHz (raw from OpenAI) - Robot must handle this format
        
        This is MUCH FASTER than process_audio_streaming() because:
        - No buffering of TTS response
        - No format conversion
        - Chunks are yielded immediately as they arrive from OpenAI
        
        Args:
            wav_bytes: Input WAV file bytes from robot
            
        Yields:
            (chunk, metadata) tuples - WAV 24kHz format
        """
        start = time.time()
        timings = {}
        
        # ⚡ PARALLEL: STT + RAG warmup
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(
                self.stt_service.transcribe_from_wav_bytes, wav_bytes
            )
            warmup_future = executor.submit(
                self.rag_service.warm_embeddings, "warmup query"
            )
            
            user_text, stt_timings = stt_future.result()
            warmup_time = warmup_future.result()
            
            timings['stt'] = stt_timings
            timings['rag_warmup'] = warmup_time
            print(f"⚡ Parallel: STT {stt_timings.get('total', 0):.2f}s | RAG warmup {warmup_time:.2f}s")
        
        # Handle empty transcription
        if not user_text:
            error_text = "Mình không nghe rõ, bạn nói lại được không?"
            yield b'', {"user_text": "", "reply": error_text, "error": True}
            
            # TRUE STREAMING - no format conversion
            for chunk in self.tts_service.synthesize_streaming(error_text, convert_format=False):
                yield chunk, {}
            
            timings['total'] = time.time() - start
            yield b'', {"timings": timings}
            return
        
        # 2. LLM with RAG - Generate response
        reply, llm_timings = self.llm_service.chat(user_text)
        timings['llm'] = llm_timings
        
        # Send metadata FIRST (before audio starts)
        yield b'', {
            "user_text": user_text, 
            "reply": reply,
            "stt_time": stt_timings.get('total', 0),
            "llm_time": llm_timings.get('total', 0),
            "format": "wav_24khz"  # Indicate raw OpenAI format
        }
        
        # 3. ⚡ TRUE STREAMING TTS - No format conversion!
        # Chunks are yielded immediately as they arrive from OpenAI
        tts_start = time.time()
        first_chunk = True
        for chunk in self.tts_service.synthesize_streaming(reply, convert_format=False):
            if first_chunk:
                print(f"⚡ First TTS chunk arrived: {time.time() - tts_start:.2f}s")
                first_chunk = False
            yield chunk, {}
        
        timings['tts'] = {'total': time.time() - tts_start}
        timings['total'] = time.time() - start
        
        yield b'', {"timings": timings}


# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_default_config() -> MeiRoboConfig:
    """Create default configuration from environment"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-api-key-here":
        raise ValueError("❌ OPENAI_API_KEY not set! Use: set OPENAI_API_KEY=sk-...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Try common candidate locations for the FAISS index folder
    candidates = [
        os.path.join(base_dir, "vector_stores", "faiss_index"),
        os.path.join(base_dir, "faiss_index"),
        os.path.join(base_dir, "vector_stores"),
    ]

    vector_db_path = None
    for c in candidates:
        if os.path.isdir(c):
            # Prefer a directory that contains index.faiss
            if os.path.exists(os.path.join(c, "index.faiss")):
                vector_db_path = c
                break

    # As a fallback, check for index.faiss files directly in a few locations
    if vector_db_path is None:
        alt_files = [
            os.path.join(base_dir, "faiss_index", "index.faiss"),
            os.path.join(base_dir, "index.faiss"),
        ]
        for f in alt_files:
            if os.path.exists(f):
                vector_db_path = os.path.dirname(f)
                break

    if vector_db_path is None:
        all_checked = candidates + alt_files
        raise ValueError(f"❌ FAISS folder not found. Checked: {all_checked}")

    # Ensure required files exist
    faiss_file = os.path.join(vector_db_path, "index.faiss")
    pkl_file = os.path.join(vector_db_path, "index.pkl")
    if not os.path.exists(faiss_file):
        raise ValueError(f"❌ Missing FAISS file: {faiss_file}")
    if not os.path.exists(pkl_file):
        raise ValueError(
            f"❌ Missing metadata file: {pkl_file}. Put index.pkl next to index.faiss or rebuild the index."
        )

    return MeiRoboConfig(
        openai_api_key=api_key,
        vector_db_path=vector_db_path
    )
