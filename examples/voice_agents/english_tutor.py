# tutor_agent_pipeline.py
import asyncio
import logging
import os
from dotenv import load_dotenv
from livekit import api
from livekit.agents import WorkerOptions, JobContext, cli, function_tool
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import (
    openai,
    silero,
    assemblyai,
    inworld,
    google,
)  # or another TTS provider
from google.genai import types
from livekit.agents import inference
from livekit.plugins import inworld
from livekit.agents.types import APIConnectOptions
from typing import Dict, Any, Optional
from livekit.plugins import openai, deepgram, cartesia, silero
from openai.types.beta.realtime.session import TurnDetection

load_dotenv()  # load env for keys

logger = logging.getLogger("tutor-pipeline")
logging.basicConfig(level=logging.INFO)


class ConversationAgent(Agent):
    def __init__(self, timeout_seconds: float = 240):
        super().__init__(
            instructions=(
                "You are an English tutor (Indian English accent). "
                "Engage the student in a natural conversation in English. "
                "Every few exchanges, call the check_timeout_and_transition function to see if it's time to move to feedback. "
                "If the function returns a transition, follow its instructions."
            )
        )
        self.timeout_seconds = timeout_seconds

    async def on_enter(self):
        self.session.userdata["transcript"] = []
        self.session.userdata["timeout_triggered"] = False
        await self.session.generate_reply(
            instructions="Hi! I'm your English tutor. Let's start talking — how are you today?"
        )
        asyncio.get_event_loop().call_later(
            30, lambda: asyncio.create_task(self._set_timeout_flag())
        )

    @function_tool
    async def check_timeout_and_transition(self):
        """Check if the 30-second timeout has been reached and transition to feedback if so."""
        if self.session.userdata.get("timeout_triggered", False):
            feedback_agent = FeedbackAgent()
            return (
                feedback_agent,
                "Time's up! Let me give you some feedback on our conversation.",
            )

        return None

    async def _set_timeout_flag(self):
        self.session.userdata["timeout_triggered"] = True


class FeedbackAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are an English tutor. Based on our conversation, "
                "give detailed feedback on grammar, phrasing, and suggest 3–5 concrete tips to improve."
            )
        )

    async def on_enter(self):
        # Optionally embed previous transcript into system prompt
        await self.session.generate_reply(
            instructions="Now I will give you feedback based on our conversation so far."
        )
        await self._delete_room()

    async def _delete_room(self):
        from livekit.agents.job import get_job_context

        job_ctx = get_job_context()
        api_client = api.LiveKitAPI(
            os.getenv("LIVEKIT_URL"),
            os.getenv("LIVEKIT_API_KEY"),
            os.getenv("LIVEKIT_API_SECRET"),
        )
        await api_client.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
        logger.info(f"{self.agent_name}: Room deleted successfully")


async def entrypoint(ctx: JobContext):
    session = AgentSession(
        # stt=assemblyai.STT(),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=inworld.TTS(model="inworld-tts-1-max", voice="Ashley"),
        vad=silero.VAD.load(),
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-09-2025",
            voice="Charon",  # Same voice for consistency
            temperature=0.6,  # Slightly lower for more consistent feedback
            conn_options=APIConnectOptions(timeout=60),
            # thinking_config=types.ThinkingConfig(
            #     include_thoughts=False,
            # ),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True,
                ),
            ),
        ),
        userdata={},
    )
    await session.start(agent=ConversationAgent(timeout_seconds=30), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
