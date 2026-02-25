# File location: 5G_Emulator_API/core_network/mrf.py
# MRF (Media Resource Function) - IMS Media Processing
# Inspired by FreeSWITCH mod_conference and media handling
# Reference: 3GPP TS 23.218 (IP Multimedia Service Control)

from fastapi import FastAPI, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any, Set
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio
import logging
import json
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MRF")

app = FastAPI(
    title="MRF - Media Resource Function",
    description="IMS MRF per 3GPP TS 23.218 - Conferencing, Announcements, Media Processing",
    version="1.0.0"
)

# ============================================================================
# Data Models - Based on FreeSWITCH conference structures
# ============================================================================

class MediaType(str, Enum):
    """Supported media types"""
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"

class ConferenceState(str, Enum):
    """Conference states"""
    PENDING = "pending"
    ACTIVE = "active"
    LOCKED = "locked"
    TERMINATED = "terminated"

class MemberState(str, Enum):
    """Conference member states"""
    CONNECTED = "connected"
    MUTED = "muted"
    DEAF = "deaf"
    ON_HOLD = "on_hold"
    DISCONNECTED = "disconnected"

class MemberFlags(BaseModel):
    """Conference member flags - based on FreeSWITCH MFLAG_*"""
    can_speak: bool = True
    can_hear: bool = True
    is_moderator: bool = False
    is_video_floor_holder: bool = False
    end_conference_on_exit: bool = False
    ghost: bool = False  # Hidden participant
    mute_detect: bool = False
    dist_dtmf: bool = True

class RtpEndpoint(BaseModel):
    """RTP endpoint information"""
    address: str
    port: int
    rtcp_port: int
    codec: str = "PCMU/8000"
    ptime: int = 20  # Packetization time in ms
    ssrc: int = 0

class MediaStream(BaseModel):
    """Media stream information"""
    stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    media_type: MediaType
    direction: str = "sendrecv"  # sendonly, recvonly, sendrecv, inactive
    local_rtp: Optional[RtpEndpoint] = None
    remote_rtp: Optional[RtpEndpoint] = None
    codec: str = "PCMU/8000"
    bandwidth: int = 64000  # bps

class ConferenceMember(BaseModel):
    """Conference member - based on FreeSWITCH conference_member_t"""
    member_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    uri: str  # SIP URI
    display_name: str = ""
    call_id: str = ""
    state: MemberState = MemberState.CONNECTED
    flags: MemberFlags = Field(default_factory=MemberFlags)
    media_streams: List[MediaStream] = []
    energy_level: int = 0  # For VAD
    volume_in_level: int = 0  # Input volume adjustment
    volume_out_level: int = 0  # Output volume adjustment
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    talk_time: int = 0  # Seconds

class Conference(BaseModel):
    """Conference room - based on FreeSWITCH conference_obj_t"""
    conference_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    uri: str = ""  # Conference SIP URI
    state: ConferenceState = ConferenceState.PENDING
    max_members: int = 100
    members: Dict[str, ConferenceMember] = {}
    # Conference settings
    rate: int = 8000  # Audio sample rate
    interval: int = 20  # Mixing interval in ms
    pin: Optional[str] = None  # Access PIN
    moderator_pin: Optional[str] = None
    record: bool = False
    recording_path: Optional[str] = None
    # Features
    mute_on_entry: bool = False
    video_floor_member: Optional[str] = None  # member_id of video floor holder
    announce_join_leave: bool = True
    comfort_noise: bool = True
    energy_threshold: int = 300  # For VAD
    created_at: datetime = Field(default_factory=datetime.utcnow)
    destroyed_at: Optional[datetime] = None

class AnnouncementType(str, Enum):
    """Types of announcements"""
    PROMPT = "prompt"
    RINGBACK = "ringback"
    BUSY = "busy"
    CONGESTION = "congestion"
    CUSTOM = "custom"

class Announcement(BaseModel):
    """Announcement/prompt definition"""
    announcement_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: AnnouncementType
    file_path: Optional[str] = None  # Path to audio file
    tts_text: Optional[str] = None  # Text-to-speech content
    language: str = "en"
    duration_ms: int = 0
    loop: bool = False

class PlaybackSession(BaseModel):
    """Active playback session"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    announcement_id: str
    target_uri: str
    call_id: str
    state: str = "playing"  # playing, paused, stopped
    position_ms: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)

class TranscodingSession(BaseModel):
    """Transcoding session for codec conversion"""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input_codec: str
    output_codec: str
    input_rtp: RtpEndpoint
    output_rtp: RtpEndpoint
    call_id: str
    state: str = "active"
    packets_transcoded: int = 0

# ============================================================================
# In-Memory Storage
# ============================================================================

class MrfStorage:
    """MRF resource storage"""

    def __init__(self):
        self.conferences: Dict[str, Conference] = {}
        self.announcements: Dict[str, Announcement] = {}
        self.playback_sessions: Dict[str, PlaybackSession] = {}
        self.transcoding_sessions: Dict[str, TranscodingSession] = {}
        self.rtp_port_pool: List[int] = list(range(30000, 32000, 2))  # RTP ports
        self._init_default_announcements()

    def _init_default_announcements(self):
        """Initialize standard announcements"""
        defaults = [
            ("welcome", AnnouncementType.PROMPT, "Welcome to the conference. Please wait."),
            ("member_joined", AnnouncementType.PROMPT, "A participant has joined."),
            ("member_left", AnnouncementType.PROMPT, "A participant has left."),
            ("conference_locked", AnnouncementType.PROMPT, "This conference is locked."),
            ("you_are_muted", AnnouncementType.PROMPT, "You are now muted."),
            ("you_are_unmuted", AnnouncementType.PROMPT, "You are now unmuted."),
            ("ringback", AnnouncementType.RINGBACK, None),
            ("busy", AnnouncementType.BUSY, None),
            ("congestion", AnnouncementType.CONGESTION, None),
        ]
        for name, ann_type, tts in defaults:
            self.announcements[name] = Announcement(
                name=name,
                type=ann_type,
                tts_text=tts
            )

    def allocate_rtp_ports(self) -> tuple:
        """Allocate RTP/RTCP port pair"""
        if not self.rtp_port_pool:
            raise RuntimeError("RTP port pool exhausted")
        rtp_port = self.rtp_port_pool.pop(0)
        return rtp_port, rtp_port + 1

    def release_rtp_ports(self, rtp_port: int):
        """Release RTP port pair back to pool"""
        if rtp_port not in self.rtp_port_pool:
            self.rtp_port_pool.append(rtp_port)
            self.rtp_port_pool.sort()

storage = MrfStorage()

# ============================================================================
# MRF Configuration
# ============================================================================

class MrfConfig:
    """MRF configuration"""
    mrf_uri: str = "sip:mrf.ims.example.com:5070"
    mrf_ip: str = "127.0.0.1"
    mrf_port: int = 5070
    # Media settings
    default_codec: str = "PCMU/8000"
    supported_codecs: List[str] = ["PCMU/8000", "PCMA/8000", "G729/8000", "AMR/8000", "AMR-WB/16000"]
    max_conferences: int = 100
    max_members_per_conference: int = 100
    # RTP
    rtp_ip: str = "127.0.0.1"
    rtp_port_start: int = 30000
    rtp_port_end: int = 32000

config = MrfConfig()

# ============================================================================
# Conference Control - Based on FreeSWITCH conference_api.c
# ============================================================================

@app.post("/conferences", response_model=Conference)
async def create_conference(
    name: str,
    max_members: int = 100,
    pin: Optional[str] = None,
    moderator_pin: Optional[str] = None,
    mute_on_entry: bool = False,
    record: bool = False
):
    """
    Create a new conference room
    Based on FreeSWITCH conference_new()
    """
    if len(storage.conferences) >= config.max_conferences:
        raise HTTPException(status_code=503, detail="Maximum conferences reached")

    conference = Conference(
        name=name,
        uri=f"sip:conference-{name}@{config.mrf_ip}",
        max_members=max_members,
        pin=pin,
        moderator_pin=moderator_pin,
        mute_on_entry=mute_on_entry,
        record=record,
        state=ConferenceState.ACTIVE
    )

    storage.conferences[conference.conference_id] = conference
    logger.info(f"[CONF] Created conference: {name} ({conference.conference_id})")

    return conference

@app.get("/conferences")
async def list_conferences():
    """List all conferences"""
    return {
        "count": len(storage.conferences),
        "conferences": [
            {
                "conference_id": c.conference_id,
                "name": c.name,
                "state": c.state.value,
                "member_count": len(c.members),
                "max_members": c.max_members,
                "created_at": c.created_at.isoformat()
            }
            for c in storage.conferences.values()
        ]
    }

@app.get("/conferences/{conference_id}")
async def get_conference(conference_id: str):
    """Get conference details"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]
    return {
        "conference_id": conf.conference_id,
        "name": conf.name,
        "uri": conf.uri,
        "state": conf.state.value,
        "members": [
            {
                "member_id": m.member_id,
                "uri": m.uri,
                "display_name": m.display_name,
                "state": m.state.value,
                "can_speak": m.flags.can_speak,
                "is_moderator": m.flags.is_moderator
            }
            for m in conf.members.values()
        ],
        "settings": {
            "max_members": conf.max_members,
            "mute_on_entry": conf.mute_on_entry,
            "record": conf.record,
            "has_pin": conf.pin is not None
        }
    }

@app.delete("/conferences/{conference_id}")
async def destroy_conference(conference_id: str):
    """
    Destroy conference and disconnect all members
    Based on FreeSWITCH conference_destroy()
    """
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]
    conf.state = ConferenceState.TERMINATED
    conf.destroyed_at = datetime.utcnow()

    # Release RTP ports for all members
    for member in conf.members.values():
        for stream in member.media_streams:
            if stream.local_rtp:
                storage.release_rtp_ports(stream.local_rtp.port)

    del storage.conferences[conference_id]
    logger.info(f"[CONF] Destroyed conference: {conf.name}")

    return {"status": "destroyed"}

@app.post("/conferences/{conference_id}/lock")
async def lock_conference(conference_id: str):
    """Lock conference - no new members can join"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    storage.conferences[conference_id].state = ConferenceState.LOCKED
    return {"status": "locked"}

@app.post("/conferences/{conference_id}/unlock")
async def unlock_conference(conference_id: str):
    """Unlock conference"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    storage.conferences[conference_id].state = ConferenceState.ACTIVE
    return {"status": "unlocked"}

# ============================================================================
# Conference Member Control - Based on FreeSWITCH conference_member.c
# ============================================================================

@app.post("/conferences/{conference_id}/members")
async def add_member(
    conference_id: str,
    uri: str,
    display_name: str = "",
    call_id: str = "",
    is_moderator: bool = False,
    pin: Optional[str] = None
):
    """
    Add member to conference
    Based on FreeSWITCH conference_add_member()
    """
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    # Check conference state
    if conf.state == ConferenceState.LOCKED:
        raise HTTPException(status_code=403, detail="Conference is locked")

    if conf.state == ConferenceState.TERMINATED:
        raise HTTPException(status_code=410, detail="Conference is terminated")

    # Check capacity
    if len(conf.members) >= conf.max_members:
        raise HTTPException(status_code=503, detail="Conference is full")

    # Verify PIN if required
    if conf.pin and pin != conf.pin:
        if not (conf.moderator_pin and pin == conf.moderator_pin):
            raise HTTPException(status_code=403, detail="Invalid PIN")
        is_moderator = True  # Moderator PIN grants moderator status

    # Allocate RTP ports
    rtp_port, rtcp_port = storage.allocate_rtp_ports()

    # Create member
    member = ConferenceMember(
        uri=uri,
        display_name=display_name or uri,
        call_id=call_id,
        flags=MemberFlags(
            is_moderator=is_moderator,
            can_speak=not conf.mute_on_entry
        ),
        media_streams=[
            MediaStream(
                media_type=MediaType.AUDIO,
                local_rtp=RtpEndpoint(
                    address=config.rtp_ip,
                    port=rtp_port,
                    rtcp_port=rtcp_port,
                    codec=config.default_codec
                )
            )
        ]
    )

    conf.members[member.member_id] = member

    logger.info(f"[CONF] Member joined: {uri} -> {conf.name}")

    # Trigger join announcement
    if conf.announce_join_leave:
        await play_announcement_to_conference(conference_id, "member_joined", exclude_member=member.member_id)

    return {
        "member_id": member.member_id,
        "conference_id": conference_id,
        "rtp_endpoint": {
            "address": config.rtp_ip,
            "port": rtp_port,
            "rtcp_port": rtcp_port,
            "codec": config.default_codec
        }
    }

@app.delete("/conferences/{conference_id}/members/{member_id}")
async def remove_member(conference_id: str, member_id: str):
    """
    Remove member from conference
    Based on FreeSWITCH conference_del_member()
    """
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    if member_id not in conf.members:
        raise HTTPException(status_code=404, detail="Member not found")

    member = conf.members[member_id]

    # Release RTP ports
    for stream in member.media_streams:
        if stream.local_rtp:
            storage.release_rtp_ports(stream.local_rtp.port)

    # Check if end conference on exit
    if member.flags.end_conference_on_exit:
        await destroy_conference(conference_id)
        return {"status": "member_removed_conference_ended"}

    del conf.members[member_id]

    logger.info(f"[CONF] Member left: {member.uri} <- {conf.name}")

    # Trigger leave announcement
    if conf.announce_join_leave:
        await play_announcement_to_conference(conference_id, "member_left")

    return {"status": "removed"}

@app.post("/conferences/{conference_id}/members/{member_id}/mute")
async def mute_member(conference_id: str, member_id: str):
    """Mute member"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    if member_id not in conf.members:
        raise HTTPException(status_code=404, detail="Member not found")

    conf.members[member_id].flags.can_speak = False
    conf.members[member_id].state = MemberState.MUTED

    return {"status": "muted"}

@app.post("/conferences/{conference_id}/members/{member_id}/unmute")
async def unmute_member(conference_id: str, member_id: str):
    """Unmute member"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    if member_id not in conf.members:
        raise HTTPException(status_code=404, detail="Member not found")

    conf.members[member_id].flags.can_speak = True
    conf.members[member_id].state = MemberState.CONNECTED

    return {"status": "unmuted"}

@app.post("/conferences/{conference_id}/members/{member_id}/deaf")
async def deaf_member(conference_id: str, member_id: str):
    """Make member deaf (can't hear others)"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    if member_id not in conf.members:
        raise HTTPException(status_code=404, detail="Member not found")

    conf.members[member_id].flags.can_hear = False
    conf.members[member_id].state = MemberState.DEAF

    return {"status": "deaf"}

@app.post("/conferences/{conference_id}/members/{member_id}/undeaf")
async def undeaf_member(conference_id: str, member_id: str):
    """Restore member hearing"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]

    if member_id not in conf.members:
        raise HTTPException(status_code=404, detail="Member not found")

    conf.members[member_id].flags.can_hear = True
    if conf.members[member_id].flags.can_speak:
        conf.members[member_id].state = MemberState.CONNECTED
    else:
        conf.members[member_id].state = MemberState.MUTED

    return {"status": "undeaf"}

@app.post("/conferences/{conference_id}/mute-all")
async def mute_all(conference_id: str, except_moderators: bool = True):
    """Mute all conference members"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]
    muted_count = 0

    for member in conf.members.values():
        if except_moderators and member.flags.is_moderator:
            continue
        member.flags.can_speak = False
        member.state = MemberState.MUTED
        muted_count += 1

    return {"muted_count": muted_count}

@app.post("/conferences/{conference_id}/unmute-all")
async def unmute_all(conference_id: str):
    """Unmute all conference members"""
    if conference_id not in storage.conferences:
        raise HTTPException(status_code=404, detail="Conference not found")

    conf = storage.conferences[conference_id]
    unmuted_count = 0

    for member in conf.members.values():
        member.flags.can_speak = True
        member.state = MemberState.CONNECTED
        unmuted_count += 1

    return {"unmuted_count": unmuted_count}

# ============================================================================
# Announcement/Prompt Playback - Based on FreeSWITCH conference_file.c
# ============================================================================

async def play_announcement_to_conference(
    conference_id: str,
    announcement_name: str,
    exclude_member: Optional[str] = None
):
    """Play announcement to conference members"""
    if announcement_name not in storage.announcements:
        return

    conf = storage.conferences.get(conference_id)
    if not conf:
        return

    ann = storage.announcements[announcement_name]
    logger.info(f"[ANN] Playing '{announcement_name}' to conference {conf.name}")

    # In production: actually send audio packets
    # For simulation: just log

@app.post("/announcements")
async def create_announcement(
    name: str,
    type: AnnouncementType = AnnouncementType.PROMPT,
    tts_text: Optional[str] = None,
    file_path: Optional[str] = None,
    language: str = "en"
):
    """Create custom announcement"""
    announcement = Announcement(
        name=name,
        type=type,
        tts_text=tts_text,
        file_path=file_path,
        language=language
    )
    storage.announcements[name] = announcement
    return announcement

@app.get("/announcements")
async def list_announcements():
    """List available announcements"""
    return {
        "count": len(storage.announcements),
        "announcements": list(storage.announcements.values())
    }

@app.post("/play")
async def play_to_uri(
    target_uri: str,
    announcement_name: str,
    call_id: str = ""
):
    """Play announcement to specific URI"""
    if announcement_name not in storage.announcements:
        raise HTTPException(status_code=404, detail="Announcement not found")

    session = PlaybackSession(
        announcement_id=storage.announcements[announcement_name].announcement_id,
        target_uri=target_uri,
        call_id=call_id
    )
    storage.playback_sessions[session.session_id] = session

    logger.info(f"[PLAY] Playing '{announcement_name}' to {target_uri}")

    return {
        "session_id": session.session_id,
        "announcement": announcement_name,
        "target": target_uri
    }

@app.delete("/play/{session_id}")
async def stop_playback(session_id: str):
    """Stop playback session"""
    if session_id not in storage.playback_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    storage.playback_sessions[session_id].state = "stopped"
    del storage.playback_sessions[session_id]

    return {"status": "stopped"}

# ============================================================================
# Transcoding - Media codec conversion
# ============================================================================

@app.post("/transcode")
async def create_transcoding_session(
    input_codec: str,
    output_codec: str,
    call_id: str,
    remote_address: str,
    remote_port: int
):
    """
    Create transcoding session for codec conversion
    Used when endpoint codecs don't match
    """
    if input_codec not in config.supported_codecs:
        raise HTTPException(status_code=400, detail=f"Unsupported input codec: {input_codec}")

    if output_codec not in config.supported_codecs:
        raise HTTPException(status_code=400, detail=f"Unsupported output codec: {output_codec}")

    # Allocate RTP ports
    in_rtp_port, in_rtcp_port = storage.allocate_rtp_ports()
    out_rtp_port, out_rtcp_port = storage.allocate_rtp_ports()

    session = TranscodingSession(
        input_codec=input_codec,
        output_codec=output_codec,
        input_rtp=RtpEndpoint(
            address=config.rtp_ip,
            port=in_rtp_port,
            rtcp_port=in_rtcp_port,
            codec=input_codec
        ),
        output_rtp=RtpEndpoint(
            address=remote_address,
            port=remote_port,
            rtcp_port=remote_port + 1,
            codec=output_codec
        ),
        call_id=call_id
    )

    storage.transcoding_sessions[session.session_id] = session

    logger.info(f"[TRANSCODE] {input_codec} -> {output_codec} for {call_id}")

    return {
        "session_id": session.session_id,
        "local_rtp": {
            "address": config.rtp_ip,
            "port": in_rtp_port,
            "codec": input_codec
        }
    }

@app.delete("/transcode/{session_id}")
async def delete_transcoding_session(session_id: str):
    """Delete transcoding session"""
    if session_id not in storage.transcoding_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = storage.transcoding_sessions[session_id]

    # Release ports
    storage.release_rtp_ports(session.input_rtp.port)

    del storage.transcoding_sessions[session_id]

    return {"status": "deleted"}

@app.get("/transcode")
async def list_transcoding_sessions():
    """List active transcoding sessions"""
    return {
        "count": len(storage.transcoding_sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "input_codec": s.input_codec,
                "output_codec": s.output_codec,
                "call_id": s.call_id,
                "packets_transcoded": s.packets_transcoded
            }
            for s in storage.transcoding_sessions.values()
        ]
    }

# ============================================================================
# SDP Manipulation
# ============================================================================

@app.post("/sdp/modify")
async def modify_sdp(
    sdp: str,
    preferred_codecs: List[str] = [],
    remove_codecs: List[str] = [],
    set_ip: Optional[str] = None,
    set_port: Optional[int] = None
):
    """
    Modify SDP for codec filtering or address manipulation
    Used for topology hiding and codec negotiation
    """
    lines = sdp.split("\r\n")
    result_lines = []

    for line in lines:
        if line.startswith("c=") and set_ip:
            # Modify connection line
            parts = line.split(" ")
            if len(parts) >= 3:
                parts[-1] = set_ip
                line = " ".join(parts)

        elif line.startswith("m=audio") and set_port:
            # Modify media port
            parts = line.split(" ")
            if len(parts) >= 2:
                parts[1] = str(set_port)
                line = " ".join(parts)

        # Filter codecs (simplified)
        if remove_codecs and line.startswith("a=rtpmap:"):
            skip = False
            for codec in remove_codecs:
                if codec.lower() in line.lower():
                    skip = True
                    break
            if skip:
                continue

        result_lines.append(line)

    return {"sdp": "\r\n".join(result_lines)}

# ============================================================================
# Statistics and Monitoring
# ============================================================================

@app.get("/statistics")
async def get_statistics():
    """Get MRF statistics"""
    total_members = sum(len(c.members) for c in storage.conferences.values())
    active_conferences = len([c for c in storage.conferences.values() if c.state == ConferenceState.ACTIVE])

    return {
        "total_conferences": len(storage.conferences),
        "active_conferences": active_conferences,
        "total_members": total_members,
        "active_playback_sessions": len(storage.playback_sessions),
        "active_transcoding_sessions": len(storage.transcoding_sessions),
        "rtp_ports_available": len(storage.rtp_port_pool)
    }

# ============================================================================
# Health and Status
# ============================================================================

@app.get("/")
async def root():
    """MRF status endpoint"""
    return {
        "nf_type": "MRF",
        "nf_name": "Media Resource Function",
        "status": "running",
        "description": "IMS MRF - Conferencing, Announcements, Media Processing",
        "version": "1.0.0",
        "mrf_uri": config.mrf_uri,
        "supported_codecs": config.supported_codecs,
        "statistics": {
            "conferences": len(storage.conferences),
            "announcements": len(storage.announcements),
            "playback_sessions": len(storage.playback_sessions),
            "transcoding_sessions": len(storage.transcoding_sessions)
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MRF", "compliance": "3GPP TS 23.228", "version": "1.0.0"}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="MRF - Media Resource Function")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("mrf"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)