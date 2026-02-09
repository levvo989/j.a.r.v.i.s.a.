from flask import Flask, request, jsonify, render_template, send_file, Response, stream_with_context
from flask import send_from_directory
from threading import Thread, Lock
from PyPDF2 import PdfReader
from pathlib import Path
import platform
import subprocess, json, os, time, re, uuid
import requests
import base64, io
import mimetypes
import sys

app = Flask(__name__, template_folder="templates")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = "config.json"
ADMIN_TOKEN_PATH = os.environ.get("ADMIN_TOKEN_PATH", "admin-token.txt")
ALICE_MAX = 1024
BOT_PROCESS = None

def _read_admin_token() -> str:
    try:
        return Path(ADMIN_TOKEN_PATH).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""
    except Exception:
        return ""

def get_admin_token() -> str:
    token = _read_admin_token()
    if token:
        return token
    return (os.environ.get("ADMIN_TOKEN", "") or "").strip()

def require_admin() -> bool:
    token = get_admin_token()
    if not token:
        return False
    header = request.headers.get("X-Admin-Token", "")
    return header == token

DEFAULT_CONFIG = {
    "flask_port": 5000,

    "cloudpub_token": "",

    "selected_model": "",
    "provider": "ollama",

    "gigachat_api_key": "",

    "gigachat_model": "GigaChat",
    
    "ollama_analysis_model": "llava:7b",
    
    "sd_enabled": True,
    "sd_url": "http://127.0.0.1:7860",
    "sd_workdir": str(PROJECT_ROOT / "sd"),
    "sd_start_cmd": r"webui-user.bat",
    "sd_default_steps": 25,
    "sd_default_width": 512,
    "sd_default_height": 512,
    "sd_timeout_sec": 900,
}

def analyze_any_file(filebytes: bytes, filename: str, usertext: str = "") -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or ""

    if mime.startswith("image/"):
        return analyze_text_with_ollama_image(
            filebytes=filebytes,
            filename=filename,
            usertext=usertext
        )

    if mime.startswith("text/") or filename.endswith((".txt", ".md", ".json")):
        try:
            text = filebytes.decode("utf-8", errors="ignore")[:15000]
        except Exception:
            text = ""
        return f"Содержимое файла:\n{text}"

    if filename.lower().endswith(".pdf"):
        return "PDF-файл. Для извлечения текста подключи pdfplumber или pymupdf."

    size_kb = len(filebytes) // 1024
    return f"Файл «{filename}», размер {size_kb} КБ. Тип: {mime or 'unknown'}."

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    changed = False
    for k, v in DEFAULT_CONFIG.items():
        if k not in cfg:
            cfg[k] = v
            changed = True
    if changed:
        save_config(cfg)
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

config = load_config()

CODER_WORKSPACE_ROOT = Path(
    os.environ.get("CODER_WORKSPACE_ROOT", str(Path(__file__).resolve().parent))
).resolve()
CODER_MAX_READ_BYTES = 2 * 1024 * 1024
CODER_HIDDEN_NAMES = {
    ".git", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"
}
CODER_CHAT_PATH = Path(__file__).resolve().parent / "coder_chats.json"
CODER_CHAT_LOCK = Lock()

def _coder_safe_rel_path(raw: str) -> str:
    rel = (raw or "").strip().replace("\\", "/")
    rel = rel.lstrip("/")
    return rel or "."

def _coder_resolve_path(raw: str) -> tuple[Path, str]:
    raw_value = (raw or "").strip()
    parsed = Path(raw_value) if raw_value else Path(".")
    if parsed.is_absolute():
        target = parsed.resolve()
    else:
        rel = _coder_safe_rel_path(raw_value)
        target = (CODER_WORKSPACE_ROOT / rel).resolve()
    if target != CODER_WORKSPACE_ROOT and CODER_WORKSPACE_ROOT not in target.parents:
        raise ValueError("Path outside workspace")
    rel_norm = str(target.relative_to(CODER_WORKSPACE_ROOT)).replace("\\", "/")
    if not rel_norm:
        rel_norm = "."
    return target, rel_norm

def _coder_ext_hint(path: str) -> str:
    ext = Path(path or "").suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".md": "markdown",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".sql": "sql",
    }.get(ext, "")

def _chat_now_ts() -> int:
    return int(time.time())

def _load_chat_store() -> dict:
    if not CODER_CHAT_PATH.exists():
        return {"chats": {}}
    try:
        with CODER_CHAT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f) or {}
        if not isinstance(data, dict):
            return {"chats": {}}
        if "chats" not in data or not isinstance(data["chats"], dict):
            data["chats"] = {}
        return data
    except Exception:
        return {"chats": {}}

def _save_chat_store(data: dict) -> None:
    tmp = CODER_CHAT_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(CODER_CHAT_PATH)

def _new_chat(title: str | None = None) -> dict:
    now = _chat_now_ts()
    chat_id = uuid.uuid4().hex
    return {
        "id": chat_id,
        "title": (title or "Новый чат").strip() or "Новый чат",
        "created_at": now,
        "last_used": now,
        "messages": []
    }


@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Admin-Token"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp
    
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        app.static_folder,
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options(_any=None):
    return ("", 204)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/mobile", methods=["GET"])
def mobile():
    return render_template("mobile.html")

@app.route("/coder", methods=["GET"])
def coder():
    return render_template("coder.html")

@app.route("/coder/config", methods=["GET"])
def coder_config():
    return jsonify({
        "workspace_root": str(CODER_WORKSPACE_ROOT),
        "provider": config.get("provider", "ollama"),
        "selected_model": config.get("selected_model", "")
    })

@app.route("/coder/workspace/set", methods=["POST"])
def coder_workspace_set():
    global CODER_WORKSPACE_ROOT
    data = request.get_json() or {}
    raw_path = (data.get("path") or "").strip()
    if not raw_path:
        return jsonify({"error": "Path is required"}), 400

    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (CODER_WORKSPACE_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        return jsonify({"error": "Path not found"}), 404
    if not candidate.is_dir():
        return jsonify({"error": "Path is not a directory"}), 400

    CODER_WORKSPACE_ROOT = candidate
    return jsonify({
        "workspace_root": str(CODER_WORKSPACE_ROOT)
    })

@app.route("/coder/fs/list", methods=["GET"])
def coder_fs_list():
    rel = request.args.get("path", ".")
    try:
        target, rel_norm = _coder_resolve_path(rel)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not target.exists():
        return jsonify({"error": "Path not found"}), 404
    if not target.is_dir():
        return jsonify({"error": "Path is not a directory"}), 400

    entries = []
    try:
        for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if child.name in CODER_HIDDEN_NAMES:
                continue
            rel_child = str(child.relative_to(CODER_WORKSPACE_ROOT)).replace("\\", "/")
            item = {
                "name": child.name,
                "path": rel_child or ".",
                "type": "dir" if child.is_dir() else "file",
            }
            if child.is_file():
                try:
                    item["size"] = child.stat().st_size
                except OSError:
                    item["size"] = 0
            entries.append(item)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "path": rel_norm,
        "entries": entries
    })

@app.route("/coder/fs/read", methods=["POST"])
def coder_fs_read():
    data = request.get_json() or {}
    rel = data.get("path", "")
    try:
        target, rel_norm = _coder_resolve_path(rel)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not target.exists() or not target.is_file():
        return jsonify({"error": "File not found"}), 404

    try:
        size = target.stat().st_size
    except OSError:
        size = 0
    if size > CODER_MAX_READ_BYTES:
        return jsonify({"error": f"File is too large (> {CODER_MAX_READ_BYTES} bytes)"}), 413

    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "path": rel_norm,
        "content": content,
        "size": size
    })

@app.route("/coder/fs/write", methods=["POST"])
def coder_fs_write():
    data = request.get_json() or {}
    rel = data.get("path", "")
    content = data.get("content", "")
    if rel in ("", ".", "/"):
        return jsonify({"error": "File path is required"}), 400

    try:
        target, rel_norm = _coder_resolve_path(rel)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if target.exists() and target.is_dir():
        return jsonify({"error": "Path is a directory"}), 400

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "saved",
        "path": rel_norm,
        "bytes": len(str(content).encode("utf-8"))
    })

@app.route("/coder/fs/new", methods=["POST"])
def coder_fs_new():
    data = request.get_json() or {}
    rel = data.get("path", "")
    kind = (data.get("kind") or "file").strip().lower()
    initial = data.get("content", "")
    if rel in ("", ".", "/"):
        return jsonify({"error": "Path is required"}), 400

    try:
        target, rel_norm = _coder_resolve_path(rel)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    try:
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(str(initial), encoding="utf-8")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "message": "created",
        "path": rel_norm,
        "type": "dir" if kind == "dir" else "file"
    })

@app.route("/coder/stream", methods=["POST"])
def coder_stream():
    data = request.get_json() or {}
    user_msg = (data.get("message") or "").strip()
    file_path = (data.get("file_path") or "untitled.txt").strip()
    file_code = data.get("code", "")
    model = (data.get("model") or config.get("selected_model") or "").strip()
    provider = (data.get("provider") or config.get("provider") or "ollama").strip()

    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    lang = _coder_ext_hint(file_path)
    prompt = f"""
You are a coding assistant inside an editor (Cursor-like).
Task from user:
{user_msg}

Current file path:
{file_path}

Current file content:
```{lang}
{file_code}
```

Response rules:
1) Start with ONLY one fenced code block containing FULL updated file content.
2) Use same language fence: {lang or 'text'}.
3) After code block, add very short explanation (1-4 bullet points).
4) Do not truncate code.
""".strip()

    def stream_text():
        if provider == "ollama":
            if not model:
                yield "Сначала выбери модель Ollama."
                return
            try:
                r = requests.post(
                    "http://127.0.0.1:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True
                    },
                    stream=True,
                    timeout=600
                )
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        chunk = obj.get("response", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        continue
            except Exception as e:
                yield f"\n[Coder stream error: {e}]"
            return

            yield generate_with_provider(prompt, timeout_sec=600)
            yield f"\n[Provider error: {e}]"

    return Response(stream_with_context(stream_text()), mimetype="text/plain; charset=utf-8")

@app.route("/coder/chats/list", methods=["GET"])
def coder_chats_list():
    with CODER_CHAT_LOCK:
        store = _load_chat_store()
        chats = list(store.get("chats", {}).values())
    chats.sort(key=lambda c: int(c.get("last_used") or 0), reverse=True)
    out = []
    for c in chats:
        out.append({
            "id": c.get("id"),
            "title": c.get("title") or "Новый чат",
            "last_used": int(c.get("last_used") or 0),
        })
    return jsonify({"chats": out})

@app.route("/coder/chats/create", methods=["POST"])
def coder_chats_create():
    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    with CODER_CHAT_LOCK:
        store = _load_chat_store()
        chat = _new_chat(title or None)
        store.setdefault("chats", {})[chat["id"]] = chat
        _save_chat_store(store)
    return jsonify({"id": chat["id"], "title": chat["title"], "created_at": chat["created_at"]})

@app.route("/coder/chats/get", methods=["GET"])
def coder_chats_get():
    chat_id = (request.args.get("id") or "").strip()
    if not chat_id:
        return jsonify({"error": "id is required"}), 400
    with CODER_CHAT_LOCK:
        store = _load_chat_store()
        chat = store.get("chats", {}).get(chat_id)
    if not chat:
        return jsonify({"error": "chat not found"}), 404
    return jsonify({
        "id": chat.get("id"),
        "title": chat.get("title") or "Новый чат",
        "created_at": int(chat.get("created_at") or 0),
        "last_used": int(chat.get("last_used") or 0),
        "messages": chat.get("messages") or [],
    })

@app.route("/coder/chats/append", methods=["POST"])
def coder_chats_append():
    data = request.get_json() or {}
    chat_id = (data.get("id") or "").strip()
    role = (data.get("role") or "").strip()
    content = (data.get("content") or "").strip()
    title_hint = (data.get("title_hint") or "").strip()
    if not chat_id or not role or not content:
        return jsonify({"error": "id, role, content are required"}), 400
    now = _chat_now_ts()
    with CODER_CHAT_LOCK:
        store = _load_chat_store()
        chats = store.setdefault("chats", {})
        chat = chats.get(chat_id)
        if not chat:
            chat = _new_chat(title_hint or None)
            chats[chat["id"]] = chat
            chat_id = chat["id"]
        chat.setdefault("messages", []).append({
            "role": role,
            "content": content,
            "ts": now
        })
        chat["last_used"] = now
        if title_hint and (not chat.get("title") or chat.get("title") == "Новый чат"):
            chat["title"] = title_hint[:80]
        _save_chat_store(store)
    return jsonify({"id": chat_id, "last_used": now, "title": chat.get("title")})

@app.route("/get-config", methods=["GET"])
def get_config():
    return jsonify({
        "provider": config.get("provider", "ollama"),
        "selected_model": config.get("selected_model", ""),

        "gigachat_model": config.get("gigachat_model"),

        "has_gigachat_key": bool(config.get("gigachat_api_key")),
        "ollama_analysis_model": config.get("ollama_analysis_model"),

    })

@app.route("/save-admin-token", methods=["POST"])
def save_admin_token():
    data = request.get_json() or {}
    new_token = (data.get("token") or "").strip()

    current_token = get_admin_token()
    if current_token and not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    try:
        Path(ADMIN_TOKEN_PATH).write_text(new_token, encoding="utf-8")
    except Exception as e:
        return jsonify({"error": f"Не удалось сохранить admin token: {e}"}), 500

    if new_token:
        msg = "Admin token сохранён"
    else:
        msg = "Admin token очищен"

    return jsonify({"message": msg, "has_admin_token": bool(new_token)})

@app.route("/list-models")
def list_models():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8")
        lines = result.stdout.strip().split("\n")
        models = [line.split()[0] for line in lines if line.strip() and not line.startswith("NAME")]
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/select-model", methods=["POST"])
def select_model():
    data = request.get_json() or {}
    config["selected_model"] = data.get("model", "")
    save_config(config)
    return jsonify({"message": "Модель выбрана"})

@app.route("/run-command", methods=["POST"])
def run_command():
    data = request.get_json() or {}
    cmd = data.get("command", "")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True, encoding="utf-8")
        output = result.stdout.strip() or result.stderr.strip()
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)})

TG_STATE = {
    "running": False
}

@app.route("/tg/status")
def tg_status():
    running = BOT_PROCESS is not None and BOT_PROCESS.poll() is None
    return jsonify({"running": running})


@app.route("/tg/config", methods=["POST"])
def tg_config():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    data = request.get_json() or {}
    tg = config.setdefault("telegram", {})

    if data.get("bot_token") is not None:
        tg["bot_token"] = data["bot_token"]

    if isinstance(data.get("admin_ids"), list):
        tg["admin_ids"] = data["admin_ids"]

    save_config(config)
    return jsonify({"message": "Telegram config saved"})

@app.route("/tg/start", methods=["POST"])
def tg_start():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    global BOT_PROCESS

    if BOT_PROCESS and BOT_PROCESS.poll() is None:
        return jsonify({"message": "Telegram bot already running"})

    try:
        BOT_PROCESS = subprocess.Popen(
            [sys.executable, "bot.py"],
            cwd=os.path.dirname(__file__),
            stdout=open("tg_bot.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        return jsonify({"message": "Telegram bot started"})
    except Exception as e:
        BOT_PROCESS = None
        return jsonify({"error": str(e)}), 500
        
@app.route("/tg/stop", methods=["POST"])
def tg_stop():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    global BOT_PROCESS

    if not BOT_PROCESS or BOT_PROCESS.poll() is not None:
        BOT_PROCESS = None
        return jsonify({"message": "Telegram bot not running"})

    BOT_PROCESS.terminate()
    BOT_PROCESS = None
    return jsonify({"message": "Telegram bot stopped"})


@app.route("/select-provider", methods=["POST"])
def select_provider():
    data = request.get_json() or {}
    provider = data.get("provider", "ollama")
    if provider not in ["ollama", "gigachat"]:
        return jsonify({"error": "Неверный провайдер"}), 400
    config["provider"] = provider
    save_config(config)
    return jsonify({"message": "Провайдер выбран"})

@app.route("/save-keys", methods=["POST"])
def save_keys():
    data = request.get_json() or {}

    if "gigachat_api_key" in data:
        config["gigachat_api_key"] = data["gigachat_api_key"]

    if "gigachat_model" in data:
        config["gigachat_model"] = data["gigachat_model"]
        
    if "ollama_analysis_model" in data:
        config["ollama_analysis_model"] = data["ollama_analysis_model"]


    save_config(config)
    return jsonify({
        "message": "Ключи/модели сохранены",
        "has_gigachat_key": bool(config.get("gigachat_api_key")),
    })

def build_prompt(user_text: str, context: str = "") -> str:
    return f"""{SYSTEM_PROMPT}

{context}

Запрос пользователя:
{user_text}
""".strip()

def generate_with_provider(user_text: str, timeout_sec: int = 600) -> str:
    provider = config.get("provider", "ollama")

    if provider == "ollama":
        model = (config.get("selected_model") or "").strip()
        if not model:
            return "Сначала выберите модель Ollama."
        try:
            r = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": model,
                    "prompt": user_text,
                    "stream": False
                    },

                timeout=timeout_sec
            )
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            return f"Ollama ошибка: {e}"

    if provider == "gigachat":
        token = config.get("gigachat_api_key")
        model = config.get("gigachat_model", "GigaChat")
        if not token:
            return "Не задан GigaChat API Key."
        try:
            r = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": model, "messages": [{"role": "user", "content": user_text}]},
                timeout=90,
                verify=False
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"GigaChat ошибка: {e}"

    return f"Неизвестный провайдер: {provider}"
            
            
def analyze_text_with_ollama_image(
    filebytes: bytes,
    filename: str = "",
    usertext: str = "",
    model: str | None = None,
    timeout_sec: int = 120
) -> str:
    m = (model or config.get("ollama_analysis_model") or "").strip()
    if not m:
        return ""

    if not filebytes:
        return ""

    prompt = (
        "Extract all readable text from the image. "
        "Return only the extracted text. "
        "Keep line breaks. "
        "Do not explain anything. "
        "Do not invent text."
    )

    try:
        img_b64 = base64.b64encode(filebytes).decode("utf-8")

        r = requests.post(
            "http://127.0.0.1:11434/api/generate",
            json={
                "model": m,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False
            },
            timeout=timeout_sec
        )
        r.raise_for_status()

        return (r.json().get("response") or "").strip()

    except Exception:
        return ""

    if provider == "gigachat":
        token = config.get("gigachat_api_key")
        model = config.get("gigachat_model", "GigaChat")
        if not token:
            return "Не задан GigaChat API Key."
        try:
            r = requests.post(
                "https://gigachat.devices.sberbank.ru/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": model, "messages": [{"role": "user", "content": user_text}]},
                timeout=90,
                verify=False
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"GigaChat ошибка: {e}"

    return f"Неизвестный провайдер: {provider}"


@app.route("/send-message", methods=["POST"])
def send_message():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Пустое сообщение."})
    output = generate_with_provider(message)
    return jsonify({"response": output})

def extract_text_from_file(filebytes: bytes, filename: str) -> str:
    name = (filename or "").lower()

    if name.endswith(".txt"):
        try:
            return filebytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(filebytes))
            parts = []
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
            return "\n\n".join(parts).strip()
        except Exception:
            return ""

    return ""

def build_file_answer_prompt(extracted_text: str, usertext: str) -> str:
    usertext = (usertext or "").strip()
    return f"""
Пользователь загрузил файл.

Извлечённый текст:
{extracted_text}

Запрос пользователя:
{usertext or "—"}

Ответь пользователю, опираясь ТОЛЬКО на текст файла.
""".strip()

@app.route("/analyze-file", methods=["POST"])
def analyze_file():
    file = request.files.get("file")
    usertext = (request.form.get("message") or "").strip()

    if not file:
        return jsonify(error="file is required"), 400

    filebytes = file.read()
    filename = file.filename or ""

    extracted = ""

    if file.mimetype.startswith("image/"):
        extracted = analyze_text_with_ollama_image(
            filebytes=filebytes,
            filename=filename,
            usertext=usertext
        )

    else:
        extracted = extract_text_from_file(filebytes, filename)

    if not extracted.strip():
        return jsonify(
            response="Не удалось извлечь текст из файла.",
            extracted=""
        )

    finalprompt = build_file_answer_prompt(extracted, usertext)
    answer = generate_with_provider(finalprompt, timeout_sec=180)

    return jsonify(response=answer, extracted=extracted)


@app.route("/send-message-stream", methods=["POST"])
def send_message_stream():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Пустое сообщение."}), 400

    provider = config.get("provider", "ollama")

    def stream_text():
        if provider == "ollama":
            model = (config.get("selected_model") or "").strip()
            if not model:
                yield "Сначала выберите модель Ollama.\n"
                return
            try:
                r = requests.post(
                    "http://127.0.0.1:11434/api/generate",

                    json={
                        "model": model,
                        "prompt": message,
                        "stream": True
                    },

                    stream=True,
                    timeout=600
                )
                r.raise_for_status()

                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        chunk = obj.get("response", "")
                        if chunk:
                            yield chunk
                    except Exception:
                        continue
            except Exception as e:
                yield f"\n[Ollama stream error: {e}]"
            return

        yield generate_with_provider(message)

    return Response(stream_with_context(stream_text()), mimetype="text/plain; charset=utf-8")

def _get_cloudpub_bin() -> str:
    base_dir = Path(__file__).resolve().parent
    clo_dir = base_dir / "clo"
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        candidates = [clo_dir / "clo.exe"]
    elif system == "darwin":
        if "arm" in machine or "aarch" in machine:
            candidates = [clo_dir / "clo_macos_arm", clo_dir / "clo-macos"]
        else:
            candidates = [clo_dir / "clo-macos", clo_dir / "clo_macos_arm"]
    elif system == "linux":
        if "arm" in machine or "aarch" in machine:
            candidates = [clo_dir / "clo_linux_arm", clo_dir / "clo-linux-arm"]
        else:
            candidates = [clo_dir / "clo_linux_x64_86", clo_dir / "clo-linux-x64"]
    else:
        candidates = []

    for candidate in candidates:
        if candidate.exists():
            if system != "windows":
                try:
                    candidate.chmod(candidate.stat().st_mode | 0o111)
                except OSError:
                    pass
            return str(candidate)

    raise FileNotFoundError(
        f"Не найден бинарник CloudPub для системы {system} ({machine}) в папке {clo_dir}"
    )


CLOUDPUB_BIN = _get_cloudpub_bin()

@app.route("/save-token", methods=["POST"])
def save_token():
    if not require_admin():
        return jsonify({"message": "Admin token required"}), 403

    data = request.get_json() or {}
    token = (data.get("token") or "").strip()

    config["cloudpub_token"] = token
    save_config(config)

    if not token:
        return jsonify({"message": "Токен очищен"})

    try:
        cmd = f'"{CLOUDPUB_BIN}" set token {token}'
        subprocess.run(cmd, shell=True, check=True)
        return jsonify({"message": "Токен CloudPub сохранён"})
    except Exception as e:
        return jsonify({"message": f"Ошибка установки токена CloudPub: {e}"})

@app.route("/start-ngrok", methods=["POST"])
def start_cloudpub():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    port = int(config.get("flask_port", 5000))

    try:
        proc = subprocess.Popen(
            [CLOUDPUB_BIN, "publish", "http", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1
        )

        def _strip_ansi(s: str) -> str:
            return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)

        def _extract_cloudpub_url(text: str) -> str | None:
            cleaned = _strip_ansi(text)
            for m in re.finditer(r"https?://\S+", cleaned):
                candidate = m.group(0).rstrip(".,;\"'")
                if not re.search(r"localhost|127\.0\.0\.1|0\.0\.0\.0", candidate):
                    return candidate
            m = re.search(r"(?<![\\w.-])([\\w.-]+\\.cloudpub\\.[\\w.-]+)(/\\S*)?", cleaned)
            if m:
                candidate = f"https://{m.group(1)}"
                if m.group(2):
                    candidate += m.group(2)
                return candidate.rstrip(".,;\"'")
            return None

        url = None
        deadline = time.time() + 20  # ждём до 20 секунд
        output_lines = []

        import queue
        import threading

        lines_queue: queue.Queue[str] = queue.Queue()

        def _reader():
            if proc.stdout is None:
                return
            for line in iter(proc.stdout.readline, ""):
                lines_queue.put(line)

        threading.Thread(target=_reader, daemon=True).start()

        while time.time() < deadline:
            try:
                line = lines_queue.get(timeout=0.2)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue

            output_lines.append(_strip_ansi(line).strip())

            candidate_url = _extract_cloudpub_url(line)
            if candidate_url:
                url = candidate_url
                break

        if not url:
            output_tail = "\n".join(output_lines[-10:]) if output_lines else ""
            if output_tail:
                fallback_url = _extract_cloudpub_url(output_tail)
                if fallback_url:
                    return jsonify({"url": fallback_url})
            return jsonify({
                "error": "CloudPub запущен, но URL не удалось получить",
                "output": output_tail
            }), 500

        return jsonify({
            "url": url
        })

    except Exception as e:
        return jsonify({
            "error": f"Ошибка запуска CloudPub: {e}"
        }), 500


SD_PROCESS = None

SD_TASKS = {}
SD_TASKS_LOCK = Lock()

def _sd_task_set(task_id, **kwargs):
    with SD_TASKS_LOCK:
        t = SD_TASKS.get(task_id, {})
        t.update(kwargs)
        SD_TASKS[task_id] = t

def _sd_task_get(task_id):
    with SD_TASKS_LOCK:
        return SD_TASKS.get(task_id)

def is_sd_running() -> bool:
    global SD_PROCESS
    if SD_PROCESS is not None and SD_PROCESS.poll() is None:
        return True
    try:
        sd_base = (config.get("sd_url") or "http://127.0.0.1:7860").rstrip("/")
        r = requests.get(f"{sd_base}/docs", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False
        
@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    
    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    payload = {
        "prompt": prompt,
        "steps": 20,
        "width": 512,
        "height": 512,
        "cfg_scale": 7,
        "sampler_name": "Euler a",
        "negative_prompt": "low quality, blurry, bad anatomy"
    }

    try:
        response = requests.post(
            "http://127.0.0.1:7860/sdapi/v1/txt2img",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("images") and len(result["images"]) > 0:
            return jsonify({"image_base64": result["images"][0]})
        else:
            return jsonify({"error": "SD WebUI не вернул изображение"}), 500
            
    except requests.exceptions.Timeout:
        return jsonify({"error": "Таймаут генерации (>2 мин)"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Ошибка SD WebUI: {str(e)}"}), 502


@app.route("/sd/status", methods=["GET"])
def sd_status():
    return jsonify({
        "running": is_sd_running(),
        "sd_url": config.get("sd_url", "http://127.0.0.1:7860"),
        "sd_workdir": config.get("sd_workdir", ""),
        "sd_start_cmd": config.get("sd_start_cmd", ""),
    })

@app.route("/sd/start", methods=["POST"])
def sd_start():
    global SD_PROCESS
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    if not config.get("sd_enabled", True):
        return jsonify({"error": "Stable Diffusion disabled in config"}), 400

    if is_sd_running():
        return jsonify({"message": "SD уже запущен"}), 200

    workdir = config.get("sd_workdir") or ""
    cmd = config.get("sd_start_cmd") or "webui-user.bat"

    if not workdir or not os.path.isdir(workdir):
        return jsonify({"error": f"sd_workdir не найден: {workdir}"}), 400

    cmdline = f'cmd.exe /c "{cmd}"'
    try:
        SD_PROCESS = subprocess.Popen(cmdline, cwd=workdir, shell=True, env=os.environ.copy())
        return jsonify({"message": "Запуск SD начат. Подожди 30–120 секунд и проверь /sd/status."})
    except Exception as e:
        SD_PROCESS = None
        return jsonify({"error": str(e)}), 500

@app.route("/sd/stop", methods=["POST"])
def sd_stop():
    global SD_PROCESS
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    if SD_PROCESS is None or SD_PROCESS.poll() is not None:
        SD_PROCESS = None
        return jsonify({"message": "SD не запущен"})

    try:
        SD_PROCESS.terminate()
        return jsonify({"message": "SD остановлен (terminate отправлен)"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/sd/models", methods=["GET"])
def sd_models():
    if not is_sd_running():
        return jsonify({"error": "SD не запущен"}), 400
    sd_base = (config.get("sd_url") or "http://127.0.0.1:7860").rstrip("/")
    try:
        r = requests.get(f"{sd_base}/sdapi/v1/sd-models", timeout=20)
        if r.status_code != 200:
            return jsonify({"error": f"SD API error {r.status_code}", "details": r.text}), 502
        return jsonify({"models": r.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

@app.route("/sd/set-model", methods=["POST"])
def sd_set_model():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403
    if not is_sd_running():
        return jsonify({"error": "SD не запущен"}), 400

    data = request.get_json() or {}
    checkpoint = (data.get("checkpoint") or "").strip()
    if not checkpoint:
        return jsonify({"error": "checkpoint is required"}), 400

    sd_base = (config.get("sd_url") or "http://127.0.0.1:7860").rstrip("/")
    try:
        r = requests.post(
            f"{sd_base}/sdapi/v1/options",
            json={"sd_model_checkpoint": checkpoint},
            timeout=30
        )
        if r.status_code != 200:
            return jsonify({"error": f"SD API error {r.status_code}", "details": r.text}), 502
        return jsonify({"message": "Модель SD выбрана"})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

def sd_generate_task(task_id: str, payload: dict):
    sd_base = (config.get("sd_url") or "http://127.0.0.1:7860").rstrip("/")
    _sd_task_set(task_id, status="running", error="", image=None)
    try:
        timeout_sec = int(config.get("sd_timeout_sec", 900))
        r = requests.post(f"{sd_base}/sdapi/v1/txt2img", json=payload, timeout=timeout_sec)
        if r.status_code != 200:
            _sd_task_set(task_id, status="error", error=f"SD API error {r.status_code}: {r.text}", image=None)
            return

        j = r.json()
        images = j.get("images") or []
        if not images:
            _sd_task_set(task_id, status="error", error="SD returned no images", image=None)
            return

        b64 = images[0]
        if "," in b64 and b64.strip().lower().startswith("data:"):
            b64 = b64.split(",", 1)[1]

        img_bytes = base64.b64decode(b64)
        _sd_task_set(task_id, status="done", error="", image=img_bytes)
    except Exception as e:
        _sd_task_set(task_id, status="error", error=f"SD task exception: {e}", image=None)

@app.route("/sd/txt2img_async", methods=["POST"])
def sd_txt2img_async():
    if not is_sd_running():
        return jsonify({"error": "SD не запущен. Запусти его в панели (/sd/start)."}), 400

    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    payload = {
        "prompt": prompt,
        "negative_prompt": (data.get("negative_prompt") or ""),
        "steps": int(data.get("steps") or config.get("sd_default_steps", 25)),
        "width": int(data.get("width") or config.get("sd_default_width", 512)),
        "height": int(data.get("height") or config.get("sd_default_height", 512)),
    }
    if data.get("seed") is not None and str(data.get("seed")).strip() != "":
        try:
            payload["seed"] = int(data["seed"])
        except Exception:
            pass

    task_id = str(uuid.uuid4())
    _sd_task_set(task_id, status="pending", error="", image=None)

    Thread(target=sd_generate_task, args=(task_id, payload), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route("/sd/task/<task_id>", methods=["GET"])
def sd_task_status(task_id):
    task = _sd_task_get(task_id)
    if not task:
        return jsonify({"error": "task not found"}), 404

    status = task.get("status", "pending")
    if status in ("pending", "running"):
        return jsonify({"status": status})

    if status == "error":
        return jsonify({"status": "error", "error": task.get("error", "")})

    img = task.get("image")
    if not img:
        return jsonify({"status": "error", "error": "empty image"}), 500

    return send_file(io.BytesIO(img), mimetype="image/png")

SD_CHECKPOINT_DIR = r"Z:\AI\J.A.R.V.I.S.A Dev CloudPub\sd_webui\models\Stable-diffusion"
ALLOWED_CKPT_EXT = {".safetensors", ".ckpt"}

try:
    from werkzeug.utils import secure_filename
except Exception:
    secure_filename = None

@app.route("/sd/upload-model", methods=["POST"])
def sd_upload_model():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403
    if secure_filename is None:
        return jsonify({"error": "werkzeug.secure_filename not available"}), 500

    if "file" not in request.files:
        return jsonify({"error": "file field is required"}), 400

    f = request.files["file"]
    if not f or not f.filename:
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_CKPT_EXT:
        return jsonify({"error": f"unsupported file type: {ext}"}), 400

    os.makedirs(SD_CHECKPOINT_DIR, exist_ok=True)
    save_path = os.path.join(SD_CHECKPOINT_DIR, filename)

    try:
        f.save(save_path)
    except Exception as e:
        return jsonify({"error": f"save failed: {e}"}), 500

    return jsonify({"message": "Модель загружена", "filename": filename, "path": save_path})

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_TASKS = {}
HF_LOCK = Lock()

def _hf_set(task_id, **kv):
    with HF_LOCK:
        t = HF_TASKS.get(task_id, {})
        t.update(kv)
        HF_TASKS[task_id] = t

def _hf_get(task_id):
    with HF_LOCK:
        return HF_TASKS.get(task_id)

def hf_download_task(task_id, repo_id, filename, revision):
    _hf_set(task_id, status="running", percent=0, downloaded=0, total=0, error="", saved_to="")
    def progress_callback(downloaded: int, total: int):
        pct = int((downloaded * 100) / total) if total else 0
        _hf_set(task_id, percent=pct, downloaded=downloaded, total=total)
    try:
        if not HF_AVAILABLE:
            raise RuntimeError("huggingface_hub not installed")
        os.makedirs(SD_CHECKPOINT_DIR, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            token=(HF_TOKEN or None),
            local_dir=SD_CHECKPOINT_DIR,
            local_dir_use_symlinks=False,
            progress_callback=progress_callback
        )
        _hf_set(task_id, status="done", percent=100, saved_to=local_path)
    except Exception as e:
        _hf_set(task_id, status="error", error=str(e))

@app.route("/hf/download_async", methods=["POST"])
def hf_download_async():
    if not require_admin():
        return jsonify({"error": "Admin token required"}), 403

    if not HF_AVAILABLE:
        return jsonify({"error": "huggingface_hub not installed (pip install huggingface_hub)"}), 400

    data = request.get_json() or {}
    repo_id = (data.get("repo_id") or "").strip()
    filename = (data.get("filename") or "").strip()
    revision = (data.get("revision") or "").strip() or None

    if not repo_id:
        return jsonify({"error": "repo_id is required"}), 400
    if not filename:
        return jsonify({"error": "filename is required (для процентов нужен конкретный файл)"}), 400

    task_id = str(uuid.uuid4())
    _hf_set(task_id, status="pending", percent=0, downloaded=0, total=0, error="", saved_to="")
    Thread(target=hf_download_task, args=(task_id, repo_id, filename, revision), daemon=True).start()
    return jsonify({"task_id": task_id})

@app.route("/hf/task/<task_id>", methods=["GET"])
def hf_task(task_id):
    t = _hf_get(task_id)
    if not t:
        return jsonify({"error": "task not found"}), 404
    return jsonify(t)

session_store = {}

CONTINUE_ONLY_REGEX = re.compile(
    r"^\s*(да|давай|окей|хорошо|продолжай|продолжи|продолжить|договаривай)\s*[.!…]*\s*$",
    re.IGNORECASE
)

ALICE_SYSTEM_PREFIX = "Отвечай кратко, не теряя суть ответа.\n\n"

def is_continue_intent(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if CONTINUE_ONLY_REGEX.match(t):
        return True
    if t.startswith("продолж") and len(t) <= 20:
        return True
    return False

def split_for_alice(text: str, chunk_size: int = ALICE_MAX):
    if not text:
        return ["(пусто)"]
    clean = " ".join(str(text).split())
    parts = []
    while len(clean) > chunk_size:
        cut = clean[:chunk_size]
        last_space = cut.rfind(" ")
        if last_space > 0:
            cut = cut[:last_space]
        parts.append(cut.strip())
        clean = clean[len(cut):].strip()
    if clean:
        parts.append(clean)
    return parts

def add_suffix_fitting(text: str, suffix: str, max_len: int = ALICE_MAX) -> str:
    text = (text or "").strip()
    if not suffix:
        return text[:max_len]
    glue = "\n\n"
    add = glue + suffix
    if len(text) + len(add) <= max_len:
        return text + add
    room = max_len - len(add)
    if room <= 0:
        return (suffix if len(suffix) <= max_len else suffix[:max_len])
    cut = text[:room]
    end_marks = [cut.rfind(ch) for ch in ".!?"]
    pos = max(end_marks)
    if pos > 50:
        cut = cut[:pos+1]
    return cut.rstrip() + add

def generate_response_async(user_text, session_id):
    output = generate_with_provider(user_text)
    parts = split_for_alice(output)
    prev = session_store.get(session_id, {})
    if prev.get("waiting") and len(parts) > 1:
        parts = [add_suffix_fitting(p, "Мне продолжить?") for p in parts[:-1]] + [parts[-1]]
    session_store[session_id] = {"full": output, "parts": parts, "index": 0}

@app.route("/alice", methods=["POST"])
def alice_webhook():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({
                "version": "1.0",
                "response": {
                    "text": "Слушаю",
                    "end_session": False
                }
            })

        session = data.get("session", {})
        request_data = data.get("request", {})
        session_id = session.get("session_id", "unknown")

        user_text = (request_data.get("original_utterance") or "").strip()

        if not user_text:
            return jsonify({
                "version": data.get("version", "1.0"),
                "session": session,
                "response": {
                    "text": "Слушаю",
                    "end_session": False
                }
            })

        info = session_store.get(session_id)
        if info and is_continue_intent(user_text):
            index = info.get("index", 0) + 1
            parts = info.get("parts", [])

            if index >= len(parts):
                return jsonify({
                    "version": data.get("version", "1.0"),
                    "session": session,
                    "response": {
                        "text": "Это был весь ответ.",
                        "end_session": False
                    }
                })

            info["index"] = index
            session_store[session_id] = info

            return jsonify({
                "version": data.get("version", "1.0"),
                "session": session,
                "response": {
                    "text": parts[index][:ALICE_MAX],
                    "end_session": False
                }
            })

        prompt = ALICE_SYSTEM_PREFIX + user_text
        output = generate_with_provider(prompt)

        parts = split_for_alice(output)
        session_store[session_id] = {
            "parts": parts,
            "index": 0
        }

        text = parts[0]
        if len(parts) > 1:
            text = add_suffix_fitting(text, "Мне продолжить?")

        return jsonify({
            "version": data.get("version", "1.0"),
            "session": session,
            "response": {
                "text": text[:ALICE_MAX],
                "end_session": False
            }
        })

    except Exception as e:
        print("ALICE ERROR:", e)
        return jsonify({
            "version": "1.0",
            "response": {
                "text": "Слушаю",
                "end_session": False
            }
        })


if __name__ == "__main__":
    print("CONFIG_PATH =", os.path.abspath(CONFIG_PATH))
    port = config.get("flask_port", 5000)
    app.run(host="0.0.0.0", port=port, debug=True)
