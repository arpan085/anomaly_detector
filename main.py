import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

# ------------------------------------------------------------
# Data models
# ------------------------------------------------------------
@dataclass
class DataPoint:
    timestamp: str
    value: float
    is_anomaly: bool = False

# Store historical data (in-memory for demo)
history: List[DataPoint] = []
# Keep a sliding window for Z-score calculation
WINDOW_SIZE = 50
window: Deque[float] = deque(maxlen=WINDOW_SIZE)

# Anomaly threshold (Z-score > 3 => anomaly)
Z_THRESHOLD = 3.0

# ------------------------------------------------------------
# Anomaly detection function (Z-score)
# ------------------------------------------------------------
def detect_anomaly_zscore(value: float) -> bool:
    if len(window) < WINDOW_SIZE:
        return False  # Not enough data yet
    arr = np.array(window)
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return False
    z = (value - mean) / std
    return abs(z) > Z_THRESHOLD

# ------------------------------------------------------------
# (Optional) Isolation Forest – uncomment to use instead
# ------------------------------------------------------------
# from sklearn.ensemble import IsolationForest
# model = IsolationForest(contamination=0.01, random_state=42)
# training_data = [...]  # you'd need to pre-train with some data
# def detect_anomaly_iforest(value: float) -> bool:
#     # Need to reshape and use model.predict()
#     pass

# ------------------------------------------------------------
# Data generator & background task
# ------------------------------------------------------------
async def generate_data(websocket_manager: "ConnectionManager"):
    """Simulate a streaming sensor and broadcast anomalies."""
    t = 0
    while True:
        # Generate a sinusoidal signal with noise and occasional spike
        base = 10 + 5 * np.sin(t / 10)
        noise = np.random.normal(0, 0.5)
        value = base + noise
        # 5% chance of a spike (anomaly)
        if random.random() < 0.05:
            value += random.uniform(10, 20)

        # Detect anomaly
        is_anomaly = detect_anomaly_zscore(value)
        # Update window (only if not anomaly? Usually you'd include all points)
        window.append(value)

        # Create datapoint
        dp = DataPoint(
            timestamp=datetime.utcnow().isoformat() + "Z",
            value=round(value, 2),
            is_anomaly=is_anomaly
        )
        history.append(dp)
        # Keep history manageable (last 1000 points)
        if len(history) > 1000:
            history.pop(0)

        # If anomaly, broadcast to all connected WebSocket clients
        if is_anomaly:
            await websocket_manager.broadcast(asdict(dp))

        t += 1
        await asyncio.sleep(0.5)  # 2 points per second

# ------------------------------------------------------------
# WebSocket connection manager
# ------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Real-time Anomaly Detector")
manager = ConnectionManager()

# Serve static files (HTML dashboard)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the dashboard HTML page."""
    with open("static/index.html", "r") as f:
        return f.read()

@app.get("/history")
async def get_history(limit: int = 100):
    """Return recent history as JSON (for initial chart load)."""
    # Return the last `limit` points, as dicts
    return [asdict(dp) for dp in history[-limit:]]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live anomaly alerts."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; we only send data on anomalies from the background task
            await websocket.receive_text()  # just to detect disconnection
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ------------------------------------------------------------
# Startup / Shutdown events
# ------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Start the background data generation task."""
    asyncio.create_task(generate_data(manager))

# ------------------------------------------------------------
# Run with: uvicorn main:app --reload
# ------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
