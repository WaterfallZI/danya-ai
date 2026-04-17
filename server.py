"""
Danya AI — Chat Server
Full ChatGPT-like experience with Groq backend
"""
from flask import Flask, request, jsonify, session, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os, re, secrets, requests, json, time, traceback

app = Flask(__name__, static_folder='.')
app.secret_key = os.environ.get('SECRET_KEY', 'danya-ai-secret-2026')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=os.environ.get('RAILWAY_ENVIRONMENT') == 'production',
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)

_db_url = os.environ.get('DATABASE_URL', 'sqlite:///danya.db')
if _db_url.startswith('postgres://'): _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config.update(SQLALCHEMY_DATABASE_URI=_db_url, SQLALCHEMY_TRACK_MODIFICATIONS=False,
    SQLALCHEMY_ENGINE_OPTIONS={'pool_pre_ping': True, 'pool_recycle': 300})
CORS(app, supports_credentials=True, origins=os.environ.get('ALLOWED_ORIGINS', '*').split(','))
db = SQLAlchemy(app)

# ── Config ────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'
ADMIN_EMAIL  = os.environ.get('ADMIN_EMAIL', 'admin@danya.ai')
ADMIN_PASS   = os.environ.get('ADMIN_PASSWORD', 'admin2026')

MODELS = {
    'danya-1.0':        {'model': 'llama-3.1-8b-instant',    'cost': 1,   'tier': 'free'},
    'danya-1.7-mj':     {'model': 'llama-3.1-8b-instant',    'cost': 1,   'tier': 'free'},
    'danya-2.5-turbo':  {'model': 'llama-3.3-70b-versatile', 'cost': 1,   'tier': 'free'},
    'danya-coala-3.7':  {'model': 'llama-3.1-8b-instant',    'cost': 1,   'tier': 'free'},
    'danya-g-4.4':      {'model': 'llama-3.3-70b-versatile', 'cost': 5,   'tier': 'free'},
    'danya-coala-4.8':  {'model': 'llama-3.3-70b-versatile', 'cost': 10,  'tier': 'free'},
    'danya-coala-5.0':  {'model': 'llama-3.3-70b-versatile', 'cost': 50,  'tier': 'pro'},
    'danya-ai-5.5':     {'model': 'llama-3.3-70b-versatile', 'cost': 80,  'tier': 'pro'},
    'danya-5.5-pro':    {'model': 'qwen-qwq-32b',            'cost': 100, 'tier': 'pro'},
    'danya-6-turbo-pro':{'model': 'qwen-qwq-32b',            'cost': 150, 'tier': 'pro'},
}

SYSTEM_PROMPTS = {
    'danya-1.0':         'You are Danya 1.0, an AI assistant by Danya AI. Be helpful, friendly and concise. Always identify as Danya 1.0.',
    'danya-1.7-mj':      'You are Danya 1.7 MJ, an AI assistant by Danya AI. Be creative and concise. Always identify as Danya 1.7 MJ.',
    'danya-2.5-turbo':   'You are Danya 2.5 Turbo, a fast powerful AI by Danya AI. Be precise and helpful. Always identify as Danya 2.5 Turbo.',
    'danya-coala-3.7':   'You are Danya Coala 3.7, a lightweight AI by Danya AI. Be quick and friendly. Always identify as Danya Coala 3.7.',
    'danya-g-4.4':       'You are Danya G 4.4, an advanced AI by Danya AI. Be smart and detailed. Always identify as Danya G 4.4.',
    'danya-coala-4.8':   'You are Danya Coala 4.8, a highly capable AI by Danya AI. Be thorough. Always identify as Danya Coala 4.8.',
    'danya-coala-5.0':   'You are Danya Coala 5.0, a premium AI by Danya AI. Be intelligent and thorough. Always identify as Danya Coala 5.0.',
    'danya-ai-5.5':      'You are Danya AI 5.5, a highly advanced AI by Danya AI. Be exceptionally precise. Always identify as Danya AI 5.5.',
    'danya-5.5-pro':     'You are Danya 5.5 Pro, a top-tier AI by Danya AI. Be exceptionally powerful. Always identify as Danya 5.5 Pro.',
    'danya-6-turbo-pro': 'You are Danya 6 Turbo Pro, THE MOST POWERFUL AI by Danya AI. Be exceptionally intelligent. Always identify as Danya 6 Turbo Pro.',
}

# ── Models ────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username      = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(256))
    credits       = db.Column(db.Integer, default=50)
    bonus_credits = db.Column(db.Integer, default=300)
    plan          = db.Column(db.String(20), default='free')
    is_banned     = db.Column(db.Boolean, default=False)
    is_admin      = db.Column(db.Boolean, default=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    last_login    = db.Column(db.DateTime)
    chats         = db.relationship('Chat', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, pw): self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return bool(self.password_hash and check_password_hash(self.password_hash, pw))

    def to_dict(self):
        total = -1 if self.plan == 'ultra' else (self.credits or 0) + (self.bonus_credits or 0)
        return {'id': self.id, 'email': self.email, 'username': self.username,
                'credits': self.credits or 0, 'bonus_credits': self.bonus_credits or 0,
                'total_credits': total, 'plan': self.plan or 'free', 'is_admin': self.is_admin or False,
                'created_at': self.created_at.isoformat() if self.created_at else None}

    def deduct(self, n=1):
        if self.plan == 'ultra': return
        if (self.bonus_credits or 0) >= n: self.bonus_credits -= n
        elif (self.credits or 0) >= n: self.credits -= n
        else:
            t = (self.bonus_credits or 0) + (self.credits or 0)
            self.bonus_credits = 0; self.credits = max(0, t - n)
        db.session.commit()

    def has_credits(self, n=1):
        if self.plan == 'ultra': return True
        return ((self.credits or 0) + (self.bonus_credits or 0)) >= n


class Chat(db.Model):
    __tablename__ = 'chats'
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title      = db.Column(db.String(200), default='New chat')
    model      = db.Column(db.String(50), default='danya-2.5-turbo')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)
    messages   = db.relationship('Message', backref='chat', lazy='dynamic', cascade='all, delete-orphan', order_by='Message.created_at')

    def to_dict(self, include_messages=False):
        d = {'id': self.id, 'title': self.title, 'model': self.model,
             'created_at': self.created_at.isoformat(), 'updated_at': self.updated_at.isoformat()}
        if include_messages:
            d['messages'] = [m.to_dict() for m in self.messages]
        return d


class Message(db.Model):
    __tablename__ = 'messages'
    id         = db.Column(db.Integer, primary_key=True)
    chat_id    = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False, index=True)
    role       = db.Column(db.String(20), nullable=False)
    content    = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {'id': self.id, 'role': self.role, 'content': self.content,
                'created_at': self.created_at.isoformat()}


with app.app_context():
    db.create_all()
    if not User.query.filter_by(email=ADMIN_EMAIL).first():
        a = User(email=ADMIN_EMAIL, username='Admin', credits=-1, bonus_credits=0, plan='ultra', is_admin=True)
        a.set_password(ADMIN_PASS); db.session.add(a); db.session.commit()


# ── Auth ──────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def d(*args, **kwargs):
        uid = session.get('user_id')
        if not uid: return jsonify({'error': 'Unauthorized'}), 401
        user = db.session.get(User, uid)
        if not user: session.clear(); return jsonify({'error': 'Unauthorized'}), 401
        if user.is_banned: return jsonify({'error': 'banned'}), 403
        return f(*args, user=user, **kwargs)
    return d


# ── Static ────────────────────────────────────────────────────────────
@app.route('/')
def index(): return send_from_directory('.', 'index.html')

@app.route('/<path:f>')
def static_files(f):
    try: return send_from_directory('.', f)
    except: return jsonify({'error': 'Not found'}), 404

@app.errorhandler(404)
def not_found(e): return jsonify({'error': 'Not found'}), 404


# ── Auth API ──────────────────────────────────────────────────────────
@app.route('/api/auth/register', methods=['POST'])
def register():
    d = request.get_json(silent=True) or {}
    username = d.get('username', '').strip()
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')
    if not username or not email or not password:
        return jsonify({'error': 'All fields required'}), 400
    if not re.match(r'^[^\s@]+@[^\s@]+\.[^\s@]+$', email):
        return jsonify({'error': 'Invalid email'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if len(username) < 2 or len(username) > 50:
        return jsonify({'error': 'Username must be 2-50 characters'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409
    try:
        user = User(email=email, username=username, credits=50, bonus_credits=300)
        user.set_password(password)
        db.session.add(user); db.session.commit()
        session.clear(); session['user_id'] = user.id; session.permanent = True
        return jsonify({'success': True, 'user': user.to_dict()})
    except Exception:
        db.session.rollback(); return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    d = request.get_json(silent=True) or {}
    email    = d.get('email', '').strip().lower()
    password = d.get('password', '')
    if not email or not password: return jsonify({'error': 'Email and password required'}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password): return jsonify({'error': 'Invalid email or password'}), 401
    if user.is_banned: return jsonify({'error': 'Account banned'}), 403
    user.last_login = datetime.utcnow(); db.session.commit()
    session.clear(); session['user_id'] = user.id; session.permanent = True
    return jsonify({'success': True, 'user': user.to_dict()})


@app.route('/api/auth/me')
@login_required
def me(user): return jsonify(user.to_dict())


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    session.clear(); return jsonify({'success': True})


# ── Chats API ─────────────────────────────────────────────────────────
@app.route('/api/chats', methods=['GET'])
@login_required
def get_chats(user):
    chats = Chat.query.filter_by(user_id=user.id).order_by(Chat.updated_at.desc()).limit(50).all()
    return jsonify([c.to_dict() for c in chats])


@app.route('/api/chats', methods=['POST'])
@login_required
def create_chat(user):
    d = request.get_json(silent=True) or {}
    chat = Chat(user_id=user.id, title=d.get('title', 'New chat'), model=d.get('model', 'danya-2.5-turbo'))
    db.session.add(chat); db.session.commit()
    return jsonify(chat.to_dict())


@app.route('/api/chats/<int:cid>', methods=['GET'])
@login_required
def get_chat(user, cid):
    chat = Chat.query.filter_by(id=cid, user_id=user.id).first()
    if not chat: return jsonify({'error': 'Not found'}), 404
    return jsonify(chat.to_dict(include_messages=True))


@app.route('/api/chats/<int:cid>', methods=['DELETE'])
@login_required
def delete_chat(user, cid):
    chat = Chat.query.filter_by(id=cid, user_id=user.id).first()
    if not chat: return jsonify({'error': 'Not found'}), 404
    db.session.delete(chat); db.session.commit()
    return jsonify({'ok': True})


@app.route('/api/chats/<int:cid>/title', methods=['POST'])
@login_required
def update_title(user, cid):
    chat = Chat.query.filter_by(id=cid, user_id=user.id).first()
    if not chat: return jsonify({'error': 'Not found'}), 404
    chat.title = (request.get_json(silent=True) or {}).get('title', chat.title)[:100]
    db.session.commit()
    return jsonify({'ok': True})


# ── Chat / AI ─────────────────────────────────────────────────────────
@app.route('/api/chats/<int:cid>/message', methods=['POST'])
@login_required
def send_message(user, cid):
    chat = Chat.query.filter_by(id=cid, user_id=user.id).first()
    if not chat: return jsonify({'error': 'Not found'}), 404

    d       = request.get_json(silent=True) or {}
    content = d.get('content', '').strip()
    model   = d.get('model', chat.model)
    stream  = d.get('stream', False)

    if not content: return jsonify({'error': 'Empty message'}), 400
    if model not in MODELS: model = 'danya-2.5-turbo'

    cfg  = MODELS[model]
    cost = cfg['cost']
    tier = cfg['tier']

    if tier == 'pro' and user.plan not in ('pro', 'max', 'ultra'):
        return jsonify({'error': 'pro_required', 'message': f'{model} requires Pro plan.'}), 403
    if not user.has_credits(cost):
        return jsonify({'error': 'no_credits', 'message': f'Need {cost} credits.'}), 402
    if not GROQ_API_KEY:
        return jsonify({'error': 'AI service not configured'}), 503

    # Save user message
    user_msg = Message(chat_id=cid, role='user', content=content)
    db.session.add(user_msg)
    chat.model = model
    chat.updated_at = datetime.utcnow()
    # Auto-title from first message
    if chat.title == 'New chat' and content:
        chat.title = content[:60] + ('…' if len(content) > 60 else '')
    db.session.commit()

    # Build messages for Groq
    history = [m.to_dict() for m in chat.messages.order_by(Message.created_at).all()]
    groq_msgs = [{'role': 'system', 'content': SYSTEM_PROMPTS.get(model, SYSTEM_PROMPTS['danya-2.5-turbo'])}]
    groq_msgs += [{'role': m['role'], 'content': m['content']} for m in history if m['role'] in ('user', 'assistant')]

    headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}
    payload = {'model': cfg['model'], 'messages': groq_msgs, 'temperature': 0.7, 'max_tokens': 4096, 'stream': stream}

    try:
        if stream:
            def generate():
                full = ''
                try:
                    resp = requests.post(GROQ_URL, headers=headers, json=payload, stream=True, timeout=60)
                    if not resp.ok:
                        yield f"data: {json.dumps({'error': f'AI error {resp.status_code}'})}\n\n"
                        return
                    for line in resp.iter_lines():
                        if not line: continue
                        line = line.decode('utf-8') if isinstance(line, bytes) else line
                        if line.startswith('data: '):
                            raw = line[6:].strip()
                            if raw == '[DONE]': break
                            try:
                                delta = json.loads(raw).get('choices', [{}])[0].get('delta', {}).get('content', '')
                                if delta:
                                    full += delta
                                    yield f"data: {json.dumps({'delta': delta})}\n\n"
                            except Exception: pass
                    # Save AI reply
                    if full:
                        ai_msg = Message(chat_id=cid, role='assistant', content=full)
                        db.session.add(ai_msg)
                        chat.updated_at = datetime.utcnow()
                        db.session.commit()
                        user.deduct(cost)
                    yield f"data: {json.dumps({'done': True, 'credits': user.to_dict()['total_credits']})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(stream_with_context(generate()), content_type='text/event-stream',
                headers={'X-Accel-Buffering': 'no', 'Cache-Control': 'no-cache'})
        else:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
            if not resp.ok:
                return jsonify({'error': f'AI error {resp.status_code}'}), resp.status_code
            reply = resp.json()['choices'][0]['message']['content']
            ai_msg = Message(chat_id=cid, role='assistant', content=reply)
            db.session.add(ai_msg)
            chat.updated_at = datetime.utcnow()
            db.session.commit()
            user.deduct(cost)
            return jsonify({'reply': reply, 'credits': user.to_dict()['total_credits']})

    except requests.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        app.logger.error(f'Chat error: {e}\n{traceback.format_exc()}')
        return jsonify({'error': str(e)}), 500


# ── User ──────────────────────────────────────────────────────────────
@app.route('/api/user/update', methods=['POST'])
@login_required
def update_user(user):
    d = request.get_json(silent=True) or {}
    if 'username' in d:
        n = d['username'].strip()
        if 2 <= len(n) <= 50: user.username = n
    if 'password' in d and len(d['password']) >= 6:
        user.set_password(d['password'])
    db.session.commit()
    return jsonify(user.to_dict())


@app.route('/api/user/delete', methods=['DELETE'])
@login_required
def delete_user(user):
    db.session.delete(user); db.session.commit(); session.clear()
    return jsonify({'success': True})


# ── Health ────────────────────────────────────────────────────────────
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'groq': bool(GROQ_API_KEY), 'models': list(MODELS.keys())})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
