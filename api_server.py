# api_server.py
# =====================================================
# FastAPI Server - STT -> LLM+RAG -> TTS Pipeline
# =====================================================

import os
import io
import time
import asyncio
import queue
import threading
from typing import Dict, Optional
from urllib.parse import quote
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from core_openai import MeiRoboPipeline, MeiRoboConfig, create_default_config

# =====================================================
# GLOBAL VARIABLES
# =====================================================

config: Optional[MeiRoboConfig] = None
pipeline: Optional[MeiRoboPipeline] = None


# =====================================================
# LIFESPAN CONTEXT MANAGER
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global config, pipeline
    
    # Startup
    print("\n" + "="*60)
    print("🚀 MeiRobo API Server - Initializing...")
    print("="*60)
    
    try:
        # Initialize config and pipeline
        config = create_default_config()
        pipeline = MeiRoboPipeline(config)
        
        print("\n" + "="*60)
        print("✅ MeiRobo API Server Ready")
        print("="*60)
        print(f"📍 Device: {config.device}")
        print(f"🧠 LLM: {config.llm_model}")
        print(f"🔊 TTS: {config.tts_model} ({config.tts_voice})")
        print(f"📚 RAG: k={config.rag_k}")
        print("="*60)
        print("\nEndpoints:")
        print("  GET  /          - Root")
        print("  GET  /health    - Health check")
        print("  POST /process_audio - Normal endpoint (WAV → WAV 16kHz)")
        print("  POST /process_audio_stream - ⚡ TRUE Streaming (WAV → WAV 24kHz, fastest!)")
        print("  POST /reset     - Reset conversation")
        print("  GET  /stats     - System stats")
        print("  GET  /docs      - Swagger API documentation")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("⚠️  Server will start but API calls will fail until fixed.\n")
        raise
    
    yield  # Server is running
    
    # Shutdown
    print("\n👋 Shutting down MeiRobo API Server...")


# =====================================================
# INITIALIZE APP
# =====================================================

app = FastAPI(
    title="MeiRobo API Server",
    description="STT -> LLM+RAG -> TTS Pipeline for Humanoid Robot",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MeiRobo API Server",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if config is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "ok",
        "device": config.device,
        "llm_model": config.llm_model,
        "tts_voice": config.tts_voice
    }


@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    """
    Main endpoint: Receives WAV -> STT -> LLM+RAG -> TTS -> Returns WAV
    
    - **audio**: WAV file (16kHz, mono, 16-bit recommended)
    - **Returns**: WAV file with speech response
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    print("\n" + "="*60)
    print(f"🎯 Request from: {audio.filename}")
    start = time.time()
    
    try:
        # Read WAV bytes
        wav_bytes = await audio.read()
        print(f"📦 Received: {len(wav_bytes)} bytes")
        
        # Process through pipeline (run in thread pool to avoid blocking)
        response_wav, timings = await asyncio.to_thread(
            pipeline.process_audio_bytes, 
            wav_bytes
        )
        
        # Get WAV properties for verification
        try:
            import wave
            with io.BytesIO(response_wav) as wav_buffer:
                with wave.open(wav_buffer, 'rb') as wav_file:
                    wav_channels = wav_file.getnchannels()
                    wav_sample_rate = wav_file.getframerate()
                    wav_sample_width = wav_file.getsampwidth()
        except Exception:
            wav_channels = 0
            wav_sample_rate = 0
            wav_sample_width = 0
        
        # Log results
        print(f"📝 User: {timings.get('user_text', 'N/A')}")
        print(f"🤖 Reply: {timings.get('reply', 'N/A')}")
        print(f"🎵 WAV Output: {wav_sample_rate}Hz, {wav_channels}-channel, {wav_sample_width*8}-bit")
        print(f"⏱️  STT: {timings.get('stt', {}).get('total', 0):.2f}s | "
              f"LLM: {timings.get('llm', {}).get('total', 0):.2f}s | "
              f"TTS: {timings.get('tts', {}).get('total', 0):.2f}s | "
              f"Total: {timings.get('total', 0):.2f}s")
        print("="*60)
        
        # Return WAV file (URL-encode Vietnamese text for headers)
        return StreamingResponse(
            io.BytesIO(response_wav),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Processing-Time": str(timings.get('total', 0)),
                "X-User-Text": quote(timings.get('user_text', ''), safe=''),
                "X-Reply-Text": quote(timings.get('reply', ''), safe=''),
                # ⚡ WAV properties for verification
                "X-WAV-Sample-Rate": str(wav_sample_rate),
                "X-WAV-Channels": str(wav_channels),
                "X-WAV-Bit-Depth": str(wav_sample_width * 8),
                # ⏱️ Timing breakdown
                "X-STT-Time": str(timings.get('stt', {}).get('total', 0)),
                "X-LLM-Time": str(timings.get('llm', {}).get('total', 0)),
                "X-TTS-Time": str(timings.get('tts', {}).get('total', 0))
            }
        )
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Error after {elapsed:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_conversation():
    """Reset conversation history"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    await asyncio.to_thread(pipeline.reset_conversation)
    return {"status": "ok", "message": "Conversation history reset"}


@app.post("/process_audio_stream")
async def process_audio_stream(audio: UploadFile = File(...)):
    """
    ⚡ TRUE STREAMING endpoint - NO buffering!
    
    **Fastest possible latency** - Audio chunks sent immediately as they arrive from OpenAI!
    
    ⚠️ Output: WAV 24kHz (raw from OpenAI) - Client must handle this format!
    
    - **audio**: WAV file (16kHz, mono, 16-bit recommended)
    - **Returns**: Streaming audio chunks (WAV format - 24kHz from OpenAI)
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    print("\n" + "="*60)
    print(f"⚡ TRUE STREAMING Request from: {audio.filename}")
    start = time.time()
    
    try:
        wav_bytes = await audio.read()
        print(f"📦 Received: {len(wav_bytes)} bytes")
        
        # ════════════════════════════════════════════════════════════
        # PHASE 1: STT + LLM (blocking) - Để có timing cho headers
        # ════════════════════════════════════════════════════════════
        from concurrent.futures import ThreadPoolExecutor
        
        # Parallel STT + RAG warmup
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(
                pipeline.stt_service.transcribe_from_wav_bytes, wav_bytes
            )
            warmup_future = executor.submit(
                pipeline.rag_service.warm_embeddings, "warmup query"
            )
            
            user_text, stt_timings = stt_future.result()
            warmup_time = warmup_future.result()
        
        stt_time = stt_timings.get('total', 0)
        print(f"⚡ STT: {stt_time:.2f}s | RAG warmup: {warmup_time:.2f}s")
        
        # Handle empty transcription
        if not user_text:
            user_text = ""
            reply_text = "Mình không nghe rõ, bạn nói lại được không?"
        else:
            # LLM with RAG
            llm_start = time.time()
            reply_text, llm_timings = await asyncio.to_thread(
                pipeline.llm_service.chat, user_text
            )
            llm_time = llm_timings.get('total', 0)
            print(f"📝 User: {user_text}")
            print(f"🤖 Reply: {reply_text}")
            print(f"⏱️  LLM: {llm_time:.2f}s")
        
        # ════════════════════════════════════════════════════════════
        # PHASE 2: TRUE STREAMING TTS - Chunks sent immediately!
        # ════════════════════════════════════════════════════════════
        first_chunk_time = None
        tts_start = time.time()
        
        async def generate():
            nonlocal first_chunk_time
            
            # ⚡ TRUE STREAMING TTS - Dùng queue để không buffer
            q = queue.Queue()
            
            def tts_producer():
                """Producer: stream TTS chunks vào queue"""
                try:
                    for chunk in pipeline.tts_service.synthesize_streaming(reply_text, convert_format=False):
                        if chunk:
                            q.put(chunk)
                except Exception as e:
                    print(f"❌ TTS error: {e}")
                finally:
                    q.put(None)  # Signal done
            
            # Start TTS producer thread
            thread = threading.Thread(target=tts_producer, daemon=True)
            thread.start()
            
            # Yield chunks as they arrive
            while True:
                chunk = await asyncio.to_thread(q.get)
                if chunk is None:
                    break
                
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start
                    print(f"⚡ First audio chunk: {first_chunk_time:.2f}s")
                
                yield chunk
            
            thread.join(timeout=5.0)
            tts_time = time.time() - tts_start
            print(f"✅ TTS streaming complete: {tts_time:.2f}s")
            print(f"✅ Total: {time.time() - start:.2f}s")
        
        return StreamingResponse(
            generate(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=response.wav",
                "X-Stream-Mode": "true",
                "X-WAV-Sample-Rate": "24000",
                "X-WAV-Channels": "1",
                "X-WAV-Bit-Depth": "16",
                # ⏱️ Timing headers (STT + LLM known before streaming)
                "X-STT-Time": str(stt_time),
                "X-LLM-Time": str(llm_time if user_text else 0),
                "X-User-Text": quote(user_text, safe=''),
                "X-Reply-Text": quote(reply_text, safe='')
            }
        )
        
    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ Error after {elapsed:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if pipeline is None or config is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "config": {
            "device": config.device,
            "llm_model": config.llm_model,
            "tts_model": config.tts_model,
            "tts_voice": config.tts_voice,
            "rag_k": config.rag_k
        },
        "has_conversation": pipeline.llm_service.last_response_id is not None
    }


@app.post("/synthesize")
async def synthesize(text: str):
    """TTS only - Normal mode (for comparison)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        start_time = time.time()
        # synthesize_to_wav_bytes returns (wav_bytes, timings)
        wav_bytes, timings = await asyncio.to_thread(
            pipeline.tts_service.synthesize_to_wav_bytes, text
        )
        tts_time = timings.get('total', time.time() - start_time)

        headers = {
            "Content-Type": "audio/wav",
            "X-TTS-Time": str(tts_time),
            "X-Text-Length": str(len(text))
        }

        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_stream")
async def synthesize_stream(text: str):
    """TTS only - Streaming mode (for comparison)"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        start_time = time.time()
        # Use the TTSService streaming generator directly (sync generator)
        def audio_generator():
            for chunk in pipeline.tts_service.synthesize_streaming(text):
                yield chunk

        tts_time = time.time() - start_time

        headers = {
            "Content-Type": "audio/wav",
            "X-TTS-Time": str(tts_time),
            "X-Text-Length": str(len(text))
        }

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers=headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "5001"))
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info")
    )

