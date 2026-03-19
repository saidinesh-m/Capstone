from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Define Models
class MessageAnalysisRequest(BaseModel):
    message: str

class ToneAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str
    emotional_tone: str
    confidence_level: str
    hidden_intent: str
    passive_aggressive: str
    power_dynamics: str
    overall_summary: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ToneAnalysisResponse(BaseModel):
    id: str
    message: str
    emotional_tone: str
    confidence_level: str
    hidden_intent: str
    passive_aggressive: str
    power_dynamics: str
    overall_summary: str
    timestamp: str

# LLM Chat instance
llm_key = os.environ['EMERGENT_LLM_KEY']

async def analyze_message_tone(message: str) -> dict:
    """Analyze message tone using LLM"""
    
    system_message = """You are TrueTone, an expert communication analyst. Analyze the given message and provide insights on:
    1. Emotional Tone (friendly, neutral, hostile, anxious, confident, etc.)
    2. Confidence Level (very confident, confident, neutral, uncertain, very uncertain)
    3. Hidden Intent (what might be the underlying message or motivation)
    4. Passive-Aggressiveness (none, subtle, moderate, high)
    5. Power Dynamics (assertive, balanced, submissive, manipulative)
    6. Overall Summary (2-3 sentences explaining the communication style)
    
    Provide clear, actionable insights that help the user understand the message better.
    Be specific and use everyday language that normal people can understand."""
    
    chat = LlmChat(
        api_key=llm_key,
        session_id=f"tone_analysis_{uuid.uuid4()}",
        system_message=system_message
    ).with_model("openai", "gpt-4o-mini")
    
    user_prompt = f"""Analyze this message:
    
    "{message}"
    
    Provide your analysis in this exact format:
    EMOTIONAL_TONE: [your analysis]
    CONFIDENCE_LEVEL: [your analysis]
    HIDDEN_INTENT: [your analysis]
    PASSIVE_AGGRESSIVE: [your analysis]
    POWER_DYNAMICS: [your analysis]
    OVERALL_SUMMARY: [your analysis]"""
    
    user_message = UserMessage(text=user_prompt)
    response = await chat.send_message(user_message)
    
    # Parse the response
    lines = response.strip().split('\n')
    analysis = {}
    
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace('_', ' ')
            analysis[key] = value.strip()
    
    return {
        'emotional_tone': analysis.get('emotional tone', 'Unable to determine'),
        'confidence_level': analysis.get('confidence level', 'Unable to determine'),
        'hidden_intent': analysis.get('hidden intent', 'Unable to determine'),
        'passive_aggressive': analysis.get('passive aggressive', 'Unable to determine'),
        'power_dynamics': analysis.get('power dynamics', 'Unable to determine'),
        'overall_summary': analysis.get('overall summary', 'Analysis unavailable')
    }

@api_router.get("/")
async def root():
    return {"message": "TrueTone API - Understand the true tone of any message"}

@api_router.post("/analyze", response_model=ToneAnalysisResponse)
async def analyze_message(request: MessageAnalysisRequest):
    """Analyze the tone and intent of a message"""
    try:
        # Analyze the message
        analysis = await analyze_message_tone(request.message)
        
        # Create analysis object
        tone_analysis = ToneAnalysis(
            message=request.message,
            emotional_tone=analysis['emotional_tone'],
            confidence_level=analysis['confidence_level'],
            hidden_intent=analysis['hidden_intent'],
            passive_aggressive=analysis['passive_aggressive'],
            power_dynamics=analysis['power_dynamics'],
            overall_summary=analysis['overall_summary']
        )
        
        # Store in database
        doc = tone_analysis.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.tone_analyses.insert_one(doc)
        
        # Return response
        return ToneAnalysisResponse(
            id=tone_analysis.id,
            message=tone_analysis.message,
            emotional_tone=tone_analysis.emotional_tone,
            confidence_level=tone_analysis.confidence_level,
            hidden_intent=tone_analysis.hidden_intent,
            passive_aggressive=tone_analysis.passive_aggressive,
            power_dynamics=tone_analysis.power_dynamics,
            overall_summary=tone_analysis.overall_summary,
            timestamp=doc['timestamp']
        )
    except Exception as e:
        logging.error(f"Error analyzing message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/history", response_model=List[ToneAnalysisResponse])
async def get_analysis_history(limit: int = 20):
    """Get analysis history"""
    try:
        analyses = await db.tone_analyses.find(
            {}, {"_id": 0}
        ).sort("timestamp", -1).limit(limit).to_list(limit)
        
        return [
            ToneAnalysisResponse(
                id=analysis['id'],
                message=analysis['message'],
                emotional_tone=analysis['emotional_tone'],
                confidence_level=analysis['confidence_level'],
                hidden_intent=analysis['hidden_intent'],
                passive_aggressive=analysis['passive_aggressive'],
                power_dynamics=analysis['power_dynamics'],
                overall_summary=analysis['overall_summary'],
                timestamp=analysis['timestamp']
            ) for analysis in analyses
        ]
    except Exception as e:
        logging.error(f"Error fetching history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()