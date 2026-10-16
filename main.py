"""AutoHeal AI: local-first monitoring and safe demonstration recovery."""
import os, time, sqlite3, json, asyncio, secrets, random, math, statistics
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
import psutil
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent
DB = Path(os.getenv('DB_PATH', str(BASE / 'autoheal.db')))
PUBLIC_DEMO = os.getenv('PUBLIC_DEMO', 'true').lower() == 'true'
ADMIN_TOKEN = os.getenv('ADMIN_TOKEN', '')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gemma3:4b')
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',')
points = deque(maxlen=240)
started = time.time()
state = {'demo_service': True, 'last_recovery': None}

def conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c

def event(level, title, detail):
    with conn() as c:
        c.execute('INSERT INTO events (ts,level,title,detail) VALUES (?,?,?,?)', (int(time.time()),level,title,detail))

def admin_required(token):
    if not ADMIN_TOKEN or not token or not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(403, 'Administrator token required')

def metrics():
    if PUBLIC_DEMO:
        t = time.time() - started
        cpu = min(99, max(5, 35 + 17*math.sin(t/23) + random.uniform(-4,4)))
        ram = min(99, max(10, 52 + 9*math.sin(t/47) + random.uniform(-2,2)))
        disk = 64.0
        return {'cpu': round(float(cpu),1), 'memory':round(float(ram),1),'disk':disk,'mode':'demo'}
    return {'cpu':psutil.cpu_percent(interval=None),'memory':psutil.virtual_memory().percent,'disk':psutil.disk_usage(os.path.abspath(os.sep)).percent,'mode':'local'}

def anomaly_info():
    if len(points) < 25:
        return {'anomaly':False,'score':None,'status':'warming_up','samples':len(points)}
    samples = list(points)
    try:
        import numpy as np
        from sklearn.ensemble import IsolationForest

        values = np.array([[p['cpu'],p['memory'],p['disk']] for p in samples],dtype=float)
        model = IsolationForest(n_estimators=60, contamination=0.08, random_state=42)
        model.fit(values[:-1])
        sample = values[-1:]
        decision = float(model.decision_function(sample)[0])
        return {'anomaly':bool(model.predict(sample)[0] == -1),'score':round(decision,3),'status':'ready','samples':len(points)}
    except (ImportError, OSError):
        features = ('cpu','memory','disk')
        deviations = []
        for feature in features:
            baseline = [point[feature] for point in samples[:-1]]
            center = statistics.median(baseline)
            spread = statistics.median(abs(value - center) for value in baseline)
            delta = abs(samples[-1][feature] - center)
            deviations.append(delta / (1.4826 * spread) if spread else (10 if delta else 0))
        score = 3.5 - max(deviations)
        return {'anomaly':score < 0,'score':round(score,3),'status':'ready','samples':len(points)}

async def collector():
    while True:
        try:
            m = metrics()
            points.append({'time':int(time.time()), **m})
            if len(points) > 25 and len(points) % 30 == 0:
                info = anomaly_info()
                if info['anomaly']:
                    event('warning','Unusual metrics','Anomaly detected in monitoring samples; investigate before taking action.')
        except Exception as exc:
            print('monitor error:',exc)
        await asyncio.sleep(3)

@asynccontextmanager
async def lifespan(app):
    with conn() as c:
        c.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts INTEGER, level TEXT, title TEXT, detail TEXT)')
    task = asyncio.create_task(collector())
    try: yield
    finally:
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass

app = FastAPI(title='AutoHeal AI API',version='0.1.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=ALLOWED_ORIGINS,allow_methods=['GET','POST'],allow_headers=['Content-Type','X-Admin-Token'])

@app.get('/api/health')
def health(): return {'ok':True,'demo':PUBLIC_DEMO,'model':OLLAMA_MODEL}

@app.get('/api/dashboard')
def dashboard():
    latest = points[-1] if points else {'time':int(time.time()),**metrics()}
    with conn() as c:
        events = [dict(row) for row in c.execute('SELECT * FROM events ORDER BY id DESC LIMIT 20')]
    return {'latest':latest,'history':list(points)[-60:],'detection':anomaly_info(), 'events':events, 'service':{'name':'Demo web service','healthy':state['demo_service']},'last_recovery':state['last_recovery']}

class ChatRequest(BaseModel):
    question: str = Field(min_length=3,max_length=1200)

@app.post('/api/chat')
async def chat(body:ChatRequest, x_admin_token:str|None=Header(default=None)):
    # Prevent anonymous visitors from consuming local model resources.
    admin_required(x_admin_token)
    latest = points[-1] if points else metrics()
    prompt = f'''You are an IT operations advisor. Do not claim certainty without evidence. Never provide executable recovery commands. Explain likely causes and safe investigation steps.\nCurrent monitoring sample: {json.dumps(latest)}\nDemo service healthy: {state['demo_service']}\nQuestion: {body.question}'''
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r = await client.post(f'{OLLAMA_URL}/api/generate',json={'model':OLLAMA_MODEL,'prompt':prompt,'stream':False})
            r.raise_for_status()
            return {'answer':r.json().get('response','No model response'),'model':OLLAMA_MODEL}
    except (httpx.HTTPError,ValueError) as exc:
        raise HTTPException(503, f'Ollama unavailable. Start Ollama and pull {OLLAMA_MODEL}. Details: {str(exc)[:160]}')

class Action(BaseModel):
    action: str

@app.post('/api/demo/action')
def demo_action(body:Action, x_admin_token:str|None=Header(default=None)):
    admin_required(x_admin_token)
    if not PUBLIC_DEMO:
        raise HTTPException(403,'Demo actions disabled outside PUBLIC_DEMO mode')
    if body.action == 'simulate_failure':
        state['demo_service'] = False
        event('critical','Demo service stopped','Simulated failure only; no real services were changed.')
        return {'ok':True,'message':'Simulated demo service failure'}
    if body.action == 'recover':
        if state['demo_service']:
            return {'ok':True,'message':'Service already healthy; no action taken'}
        state['demo_service'] = True
        state['last_recovery'] = int(time.time())
        event('success','Demo service recovered','Approved simulated recovery completed; demo health verified.')
        return {'ok':True,'message':'Simulated recovery successful; health verified'}
    raise HTTPException(400,'Unsupported action')
