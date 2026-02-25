# File location: 5G_Emulator_API/core_network/udr.py
# File location: 5G_Emulator_API/core_network/udr.py
# File location: 5G_Emulator_API/core_network/udr.py
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import uvicorn

app = FastAPI()

class UserData(BaseModel):
    imsi: str
    key: str

def init_db():
    conn = sqlite3.connect('udr.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            imsi TEXT PRIMARY KEY,
            key TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

@app.post("/register_user")
def register_user(user: UserData):
    conn = sqlite3.connect('udr.db')
    c = conn.cursor()
    c.execute('INSERT INTO users (imsi, key) VALUES (?, ?)', (user.imsi, user.key))
    conn.commit()
    conn.close()
    return {"message": "User registered successfully"}

@app.get("/get_user/{imsi}")
def get_user(imsi: str):
    conn = sqlite3.connect('udr.db')
    c = conn.cursor()
    c.execute('SELECT key FROM users WHERE imsi = ?', (imsi,))
    user = c.fetchone()
    conn.close()
    if user:
        return {"imsi": imsi, "key": user[0]}
    else:
        return {"message": "User not found"}

@app.get("/health")
def health_check():
    """Health check endpoint for UDR - TS 29.504"""
    return {
        "status": "healthy",
        "service": "UDR",
        "compliance": "3GPP TS 29.504",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config.ports import get_port

    parser = argparse.ArgumentParser(description="UDR - Unified Data Repository")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=get_port("udr"), help="Port to bind to")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)