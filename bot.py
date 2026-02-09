import os
import json
import time
import logging
import threading
import requests
import telebot
from telebot.apihelper import ApiTelegramException
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
LOG_PATH = os.path.join(BASE_DIR, "tg_bot.log")

BACKEND_URL = "http://127.0.0.1:5000"


logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)

logging.info("Bot process starting")


if not os.path.exists(CONFIG_PATH):
    raise RuntimeError("config.json not found")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

tg_cfg = config.get("telegram") or {}
TG_TOKEN = (tg_cfg.get("bot_token") or "").strip()
ADMIN_IDS = set(tg_cfg.get("admin_ids") or [])

if not TG_TOKEN or ":" not in TG_TOKEN:
    raise RuntimeError("Invalid Telegram bot token in config.json")


bot = telebot.TeleBot(TG_TOKEN, parse_mode="HTML")


def is_admin(user_id: int) -> bool:
    return not ADMIN_IDS or user_id in ADMIN_IDS

def safe_edit(chat_id, message_id, text, reply_markup=None):
    try:
        bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text[:4096],
            reply_markup=reply_markup
        )
    except ApiTelegramException:
        pass


def ask_backend(prompt: str) -> str:
    try:
        r = requests.post(
            f"{BACKEND_URL}/send-message",
            json={"message": prompt},
            timeout=600
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()
    except Exception as e:
        logging.exception("Backend error")
        return f"Backend error:\n{e}"

def analyze_file_backend(file_bytes: bytes, filename: str, prompt: str) -> str:
    try:
        files = {
            "file": (filename or "file", file_bytes)
        }
        data = {
            "message": (prompt or "").strip()
        }
        r = requests.post(
            f"{BACKEND_URL}/analyze-file",
            files=files,
            data=data,
            timeout=600
        )
        r.raise_for_status()
        payload = r.json()
        return (payload.get("response") or "").strip()
    except Exception as e:
        logging.exception("Backend analyze-file error")
        return f"Backend error:\n{e}"

def analyze_file_backend(file_bytes: bytes, filename: str, prompt: str) -> str:
    try:
        files = {
            "file": (filename or "file", file_bytes)
        }
        data = {
            "message": (prompt or "").strip()
        }
        r = requests.post(
            f"{BACKEND_URL}/analyze-file",
            files=files,
            data=data,
            timeout=600
        )
        r.raise_for_status()
        payload = r.json()
        return (payload.get("response") or "").strip()
    except Exception as e:
        logging.exception("Backend analyze-file error")
        return f"Backend error:\n{e}"

def analyze_file_backend(file_bytes: bytes, filename: str, prompt: str) -> str:
    try:
        files = {
            "file": (filename or "file", file_bytes)
        }
        data = {
            "message": (prompt or "").strip()
        }
        r = requests.post(
            f"{BACKEND_URL}/analyze-file",
            files=files,
            data=data,
            timeout=600
        )
        r.raise_for_status()
        payload = r.json()
        return (payload.get("response") or "").strip()
    except Exception as e:
        logging.exception("Backend analyze-file error")
        return f"Backend error:\n{e}"

def get_backend_config():
    r = requests.get(f"{BACKEND_URL}/get-config", timeout=5)
    r.raise_for_status()
    return r.json()

def set_provider(provider: str):
    requests.post(
        f"{BACKEND_URL}/select-provider",
        json={"provider": provider},
        timeout=5
    )

def set_model(model: str):
    requests.post(
        f"{BACKEND_URL}/select-model",
        json={"model": model},
        timeout=5
    )

def list_models():
    r = requests.get(f"{BACKEND_URL}/list-models", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])

def settings_root_kb():
    kb = InlineKeyboardMarkup(row_width=1)
    kb.add(
        InlineKeyboardButton("🧠 Провайдер", callback_data="settings:provider"),
        InlineKeyboardButton("📦 Модель", callback_data="settings:model")
    )
    return kb


def provider_kb(current):
    kb = InlineKeyboardMarkup(row_width=2)
    providers = ["ollama", "gigachat"]

    for p in providers:
        title = f"✅ {p}" if p == current else p
        kb.add(InlineKeyboardButton(title, callback_data=f"prov:{p}"))

    kb.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:root"))
    return kb


def model_kb(models, current):
    kb = InlineKeyboardMarkup(row_width=1)
    for m in models:
        title = f"✅ {m}" if m == current else m
        kb.add(InlineKeyboardButton(title, callback_data=f"model:{m}"))

    kb.add(InlineKeyboardButton("⬅️ Назад", callback_data="settings:root"))
    return kb



def backend_status() -> str:
    try:
        r = requests.get(f"{BACKEND_URL}/get-config", timeout=5)
        r.raise_for_status()
        cfg = r.json()
        return (
            "Current backend settings:\n\n"
            f"Provider: {cfg.get('provider')}\n"
            f"Model: {cfg.get('selected_model') or '—'}"
        )
    except Exception:
        return "Backend unavailable"




@bot.callback_query_handler(func=lambda c: True)
def on_callback(c):
    if not is_admin(c.from_user.id):
        bot.answer_callback_query(c.id, "Нет доступа")
        return

    try:
        cfg = get_backend_config()

        if c.data == "settings:root":
            safe_edit(c.message.chat.id, c.message.message_id,
                      "⚙️ Настройки", settings_root_kb())

        elif c.data == "settings:provider":
            safe_edit(
                c.message.chat.id,
                c.message.message_id,
                "🧠 Выбери провайдер",
                provider_kb(cfg.get("provider"))
            )

        elif c.data.startswith("prov:"):
            provider = c.data.split(":", 1)[1]
            set_provider(provider)
            safe_edit(
                c.message.chat.id,
                c.message.message_id,
                f"✅ Провайдер: <b>{provider}</b>",
                settings_root_kb()
            )

        elif c.data == "settings:model":
            models = list_models()
            safe_edit(
                c.message.chat.id,
                c.message.message_id,
                "📦 Выбери модель",
                model_kb(models, cfg.get("selected_model"))
            )

        elif c.data.startswith("model:"):
            model = c.data.split(":", 1)[1]
            set_model(model)
            safe_edit(
                c.message.chat.id,
                c.message.message_id,
                f"✅ Модель: <b>{model}</b>",
                settings_root_kb()
            )

        bot.answer_callback_query(c.id)

    except Exception as e:
        bot.answer_callback_query(c.id, "Ошибка")
        logging.exception("Settings callback error")

@bot.message_handler(commands=["status"])
def cmd_status(m):
    if not is_admin(m.from_user.id):
        return
    bot.reply_to(m, backend_status())

@bot.message_handler(content_types=["photo", "document"])
def handle_file(m):
    if not is_admin(m.from_user.id):
        return

    prompt = (m.caption or "").strip()
    msg = bot.reply_to(m, "Анализирую файл...")
    stop_event = threading.Event()

    def progress_worker():
        percent = 0
        while not stop_event.is_set():
            percent = min(percent + 10, 90)
            safe_edit(m.chat.id, msg.message_id, f"Анализ файла: {percent}%")
            stop_event.wait(2.0)

    progress_thread = threading.Thread(target=progress_worker, daemon=True)
    progress_thread.start()

    try:
        if m.content_type == "photo":
            photo = m.photo[-1]
            file_info = bot.get_file(photo.file_id)
            file_bytes = bot.download_file(file_info.file_path)
            filename = os.path.basename(file_info.file_path) or "photo.jpg"
        else:
            doc = m.document
            file_info = bot.get_file(doc.file_id)
            file_bytes = bot.download_file(file_info.file_path)
            filename = doc.file_name or os.path.basename(file_info.file_path) or "file"

        answer = analyze_file_backend(file_bytes, filename, prompt)
        safe_edit(m.chat.id, msg.message_id, "Анализ файла: 100%")
        safe_edit(m.chat.id, msg.message_id, answer or "Пустой ответ.")
    except Exception as e:
        logging.exception("File handler error")
        safe_edit(m.chat.id, msg.message_id, f"Ошибка анализа файла:\n{e}")
    finally:
        stop_event.set()

@bot.message_handler(func=lambda m: True)
def chat(m):
    if not is_admin(m.from_user.id):
        return

    text = (m.text or "").strip()
    if not text:
        return

    if text.startswith("/settings"):
        bot.send_message(
            m.chat.id,
            "⚙️ Настройки",
            reply_markup=settings_root_kb()
        )
        return

    if text.startswith("/status"):
        bot.reply_to(m, backend_status())
        return

    msg = bot.reply_to(m, "Думаю...")

    start = time.time()
    answer = ask_backend(text)
    elapsed = time.time() - start

    answer += f"\n\nTime: {elapsed:.1f}s"
    safe_edit(m.chat.id, msg.message_id, answer)




try:
    bot.remove_webhook()
except Exception:
    pass

logging.info("Telegram bot started")
bot.infinity_polling(
    skip_pending=True,
    timeout=30,
    long_polling_timeout=30
)