import queue
import sys
import threading
from datetime import timedelta
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st
import psutil
import GPUtil
import requests

from gui.services.duration_fmt import format_duration_ru
from gui.services.file_processing import (
    PIPELINE_STAGE_DESCRIPTIONS,
    default_enabled_stages,
    run_conversion_job,
)
from gui.services.gui_settings import load_gui_settings, save_gui_settings
from llmmd_core.config import PROJECT_VERSION, load_launcher_config
from llmmd_core.dependencies import missing_imports
from llmmd_core.docker import docker_status, qdrant_http_ready
from llmmd_core.processes import (
    ManagedProcess,
    run_cli_capture,
    start_cli_process,
    stop_process,
    tail_text,
)

from rag_service.config import Settings
from rag_service.ui_settings_store import load_settings_overlay, save_settings_to_file

LAUNCHER_CONFIG = load_launcher_config()

# ---- ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ ----
if "gui_settings" not in st.session_state:
    st.session_state.gui_settings = load_gui_settings()

if "rag_settings" not in st.session_state:
    st.session_state.rag_settings = load_settings_overlay(Settings())

if "job_events" not in st.session_state:
    st.session_state.job_events = queue.Queue()

if "job_thread" not in st.session_state:
    st.session_state.job_thread = None

if "job_log" not in st.session_state:
    st.session_state.job_log = []

if "job_stage_events" not in st.session_state:
    st.session_state.job_stage_events = []

if "job_progress" not in st.session_state:
    st.session_state.job_progress = 0

if "job_progress_max" not in st.session_state:
    st.session_state.job_progress_max = 1

if "job_status" not in st.session_state:
    st.session_state.job_status = "Ожидание..."

if "job_done_message" not in st.session_state:
    st.session_state.job_done_message = ""

if "cancel_event" not in st.session_state:
    st.session_state.cancel_event = threading.Event()

if "pause_event" not in st.session_state:
    st.session_state.pause_event = threading.Event()

if "managed_processes" not in st.session_state:
    st.session_state.managed_processes = {}

if "last_cli_result" not in st.session_state:
    st.session_state.last_cli_result = None


def update_gui_setting(key, val):
    st.session_state.gui_settings[key] = val
    save_gui_settings(st.session_state.gui_settings)


def get_gui_setting(key, default):
    return st.session_state.gui_settings.get(key, default)


def get_rag_api_base_url():
    s = st.session_state.rag_settings
    return f"http://{s.api_host}:{s.api_port}"


def _root() -> Path:
    return LAUNCHER_CONFIG.paths.root


def _managed() -> dict[str, ManagedProcess]:
    return st.session_state.managed_processes


def _process(name: str) -> ManagedProcess | None:
    proc = _managed().get(name)
    if proc is not None and not isinstance(proc, ManagedProcess):
        _managed().pop(name, None)
        return None
    return proc


def _start_process(name: str, args: list[str]) -> None:
    current = _process(name)
    if current is not None and current.is_running:
        st.warning(f"Процесс `{name}` уже запущен: PID {current.pid}")
        return
    _managed()[name] = start_cli_process(name, args, root=_root())
    st.success(f"Запущено: `{name}`")


def _stop_process(name: str) -> None:
    proc = _process(name)
    if proc is None:
        st.info(f"Процесс `{name}` не запускался из этой UI-сессии.")
        return
    stop_process(proc)
    st.success(f"Остановлено: `{name}`")


def _render_process_card(name: str, title: str, args: list[str], *, log_lines: int = 80) -> None:
    proc = _process(name)
    running = proc is not None and proc.is_running
    status = "работает" if running else ("остановлен" if proc else "не запущен из UI")
    st.markdown(f"**{title}**")
    st.caption(f"`python llmmd.py {' '.join(args)}`")
    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.metric("Статус", status)
    with cols[1]:
        st.metric("PID", str(proc.pid) if proc else "-")
    with cols[2]:
        c_start, c_stop = st.columns(2)
        with c_start:
            if st.button("Запустить", key=f"start_{name}", use_container_width=True, disabled=running):
                _start_process(name, args)
                st.rerun()
        with c_stop:
            if st.button("Остановить", key=f"stop_{name}", use_container_width=True, disabled=not running):
                _stop_process(name)
                st.rerun()
    if proc is not None and st.checkbox(f"Показать лог {title}", key=f"show_log_{name}"):
        st.code(tail_text(proc.log_path, lines=log_lines) or "Лог пуст.", language="log")


def _run_short_cli(args: list[str], *, timeout_s: float = 120.0) -> None:
    with st.spinner(f"Выполняется: python llmmd.py {' '.join(args)}"):
        try:
            res = run_cli_capture(args, root=_root(), timeout_s=timeout_s)
        except Exception as e:
            st.session_state.last_cli_result = {
                "args": args,
                "returncode": 1,
                "output": f"{type(e).__name__}: {e}",
            }
            return
    st.session_state.last_cli_result = {
        "args": args,
        "returncode": res.returncode,
        "output": res.stdout,
    }


def _render_last_cli_result() -> None:
    res = st.session_state.get("last_cli_result")
    if not res:
        return
    args = " ".join(res["args"])
    rc = res["returncode"]
    if rc == 0:
        st.success(f"`python llmmd.py {args}` завершилась с кодом 0")
    else:
        st.error(f"`python llmmd.py {args}` завершилась с кодом {rc}")
    st.code(res["output"] or "", language="text")


def _remember_stage_event(value: dict) -> None:
    if not isinstance(value, dict):
        return
    rows = list(st.session_state.job_stage_events)
    stage_id = value.get("stage_id")
    if stage_id:
        for i, row in enumerate(rows):
            if row.get("stage_id") == stage_id:
                rows[i] = {**row, **value}
                st.session_state.job_stage_events = rows
                return
    rows.append(value)
    st.session_state.job_stage_events = rows[-300:]


def _render_stage_history() -> None:
    rows = st.session_state.get("job_stage_events") or []
    if not rows:
        st.info("История этапов появится после запуска обработки.")
        return
    status_map = {
        "running": "выполняется",
        "done": "готово",
        "failed": "ошибка",
        "skipped": "пропущено",
    }
    table = []
    for row in rows:
        duration = row.get("duration_s")
        duration_text = format_duration_ru(duration) if isinstance(duration, (int, float)) else ""
        table.append(
            {
                "Статус": status_map.get(row.get("status"), row.get("status") or ""),
                "Файл": row.get("file_name") or "",
                "Этап": row.get("label") or row.get("stage") or "",
                "Модель": row.get("model") or "",
                "Время": duration_text,
                "Детали": row.get("details") or "",
            }
        )
    st.dataframe(table, use_container_width=True, hide_index=True)


def apply_ocr_gui_settings_blob(blob: dict) -> None:
    for key, val in blob.items():
        update_gui_setting(key, val)


def poll_events():
    while not st.session_state.job_events.empty():
        try:
            event, value = st.session_state.job_events.get_nowait()
            if event == "log":
                st.session_state.job_log.append(value)
            elif event == "progress_max":
                st.session_state.job_progress_max = max(1, int(value))
            elif event == "progress":
                st.session_state.job_progress = int(value)
            elif event == "current":
                st.session_state.job_status = f"Обработка: {value}"
            elif event == "stage":
                _remember_stage_event(value)
                if isinstance(value, dict):
                    status = value.get("status")
                    label = value.get("label") or value.get("stage") or "этап"
                    file_name = value.get("file_name") or ""
                    if status == "running":
                        st.session_state.job_status = f"{label}: {file_name}".strip(": ")
                    elif status == "done" and value.get("duration_s") is not None:
                        st.session_state.job_status = f"{label}: готово за {value.get('duration_s')} c"
            elif event in ("ocr", "ocr_page"):
                st.session_state.job_status = f"Прогресс OCR: {value}"
            elif event == "status":
                st.session_state.job_status = str(value)
            elif event == "done":
                st.session_state.job_done_message = str(value)
                st.session_state.job_status = str(value)
        except queue.Empty:
            break


@st.fragment(run_every=timedelta(seconds=2))
def _render_job_progress_section(*, log_tail: int = 25, show_stage_history: bool = True) -> None:
    poll_events()
    thread = st.session_state.job_thread
    alive = thread is not None and thread.is_alive()
    log = st.session_state.job_log
    if not alive and not log:
        return

    if alive:
        st.progress(st.session_state.job_progress / max(1, st.session_state.job_progress_max))
        st.text(st.session_state.job_status)
    else:
        done_msg = (st.session_state.job_done_message or "").strip()
        if done_msg:
            st.success(f"Задача завершена: {done_msg}")
        else:
            st.success("Задача завершена или отменена!")

    if show_stage_history and (alive or st.checkbox("Показать историю этапов", key="show_stage_history")):
        st.markdown("**История этапов**")
        _render_stage_history()
    st.code("\n".join(log[-log_tail:]), language="log")


# ---- СЕКЦИИ ОДНОСТРАНИЧНОГО UI ----

def render_system_section():
    col1, col2 = st.columns(2)
    with col1:
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        st.metric("CPU", f"{cpu_percent}%")
        st.progress(cpu_percent / 100)
        st.metric(
            "RAM",
            f"{mem.percent}% ({mem.used / 1024**3:.1f} GB / {mem.total / 1024**3:.1f} GB)",
        )
        st.progress(mem.percent / 100)

    with col2:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for idx, gpu in enumerate(gpus):
                    st.metric(
                        f"GPU {idx}: {gpu.name}",
                        f"{gpu.load * 100:.1f}% (VRAM: {gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB)",
                    )
                    st.progress(gpu.load)
            else:
                st.info("GPU не найдены.")
        except Exception as e:
            st.error(f"Не удалось получить информацию о GPU: {e}")

        api_url = get_rag_api_base_url()
        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200 and resp.json().get("ok"):
                data = resp.json()
                st.success(
                    f"RAG: OK (гибрид: {data.get('hybrid')}, dim: {data.get('dense_dim')})"
                )
            else:
                st.warning("RAG запущен, но есть проблемы.")
        except Exception:
            st.error("RAG недоступен. Запустите в блоке «Сервисы».")


def render_services_section():
    tab_qdrant, tab_processes = st.tabs(["Qdrant", "RAG API"])

    with tab_qdrant:
        cfg = LAUNCHER_CONFIG.qdrant
        if cfg is None:
            st.error("Qdrant не настроен в config/llmmd.yaml.")
            return
        ds = docker_status(root=_root())
        q_ok, q_msg = qdrant_http_ready(cfg, timeout_s=0.75)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Docker", "ok" if ds.ok else "недоступен")
        with c2:
            st.metric("Qdrant HTTP", "ready" if q_ok else "недоступен")
        with c3:
            st.metric("URL", cfg.url)
        if not ds.ok:
            st.warning(ds.message)
        if not q_ok:
            st.info(q_msg)
        cols = st.columns(5)
        actions = [
            ("up", ["qdrant", "up"], 180.0),
            ("status", ["qdrant", "status"], 30.0),
            ("logs", ["qdrant", "logs"], 60.0),
            ("restart", ["qdrant", "restart"], 240.0),
            ("down", ["qdrant", "down"], 120.0),
        ]
        for col, (label, args, timeout) in zip(cols, actions):
            with col:
                if st.button(label, key=f"qdrant_{label}", use_container_width=True):
                    _run_short_cli(args, timeout_s=timeout)
                    st.rerun()
        _render_last_cli_result()

    with tab_processes:
        _render_process_card("rag", "RAG API", ["rag"])


def render_rag_config_section():
    s = st.session_state.rag_settings

    s.qdrant_url = st.text_input("Qdrant URL", value=s.qdrant_url)
    s.qdrant_api_key = st.text_input("Qdrant API ключ", type="password", value=s.qdrant_api_key or "")
    s.collection_name = st.text_input("Имя коллекции", value=s.collection_name)
    s.corpus_root = st.text_input(
        "Папка с документами",
        value=str(s.corpus_root) if s.corpus_root else "",
    )

    c1, c2 = st.columns(2)
    with c1:
        s.embedding_model = st.text_input("Модель эмбеддингов", value=s.embedding_model)
        s.embedding_device = st.text_input("Устройство (cpu/cuda)", value=s.embedding_device or "")
    with c2:
        s.enable_hybrid = st.checkbox("Гибридный поиск (BM25 + Dense)", value=s.enable_hybrid)
        s.rag_dedupe_sources = st.checkbox("Убирать дубли источников", value=s.rag_dedupe_sources)

    c3, c4 = st.columns(2)
    with c3:
        s.rag_context_max_chars = st.number_input(
            "Макс. символов контекста", 1000, 200000, value=s.rag_context_max_chars
        )
    with c4:
        s.rag_source_max_chars = st.number_input(
            "Макс. символов источника", 500, 100000, value=s.rag_source_max_chars
        )

    st.subheader("LLM-провайдеры")
    s.default_rag_llm_provider = st.selectbox(
        "Провайдер по умолчанию",
        ["auto", "lm_studio", "openai", "anthropic"],
        index=["auto", "lm_studio", "openai", "anthropic"].index(s.default_rag_llm_provider),
    )
    s.lm_studio_base_url = st.text_input("LM Studio URL", value=s.lm_studio_base_url)
    s.lm_studio_api_key = st.text_input(
        "LM Studio API ключ", type="password", value=s.lm_studio_api_key or ""
    )
    s.lm_studio_rag_model = st.text_input("Модель RAG (LM Studio)", value=s.lm_studio_rag_model)
    s.default_llm_model = st.text_input("Модель OpenAI", value=s.default_llm_model)
    s.openai_api_key = st.text_input("OpenAI API ключ", type="password", value=s.openai_api_key or "")
    s.anthropic_api_key = st.text_input(
        "Anthropic API ключ", type="password", value=s.anthropic_api_key or ""
    )
    s.anthropic_model = st.text_input("Модель Anthropic", value=s.anthropic_model)

    if st.button("Сохранить настройки RAG", type="primary", key="save_rag"):
        if not s.corpus_root:
            s.corpus_root = None
        if not s.embedding_device:
            s.embedding_device = None
        if not s.qdrant_api_key:
            s.qdrant_api_key = None
        if not s.lm_studio_api_key:
            s.lm_studio_api_key = None
        if not s.openai_api_key:
            s.openai_api_key = None
        if not s.anthropic_api_key:
            s.anthropic_api_key = None
        save_settings_to_file(s)
        api_url = get_rag_api_base_url()
        try:
            r = requests.put(
                f"{api_url}/v1/settings",
                json={"patch": s.model_dump(mode="json")},
                timeout=30,
            )
            if r.status_code == 200:
                st.success("Настройки сохранены и применены.")
            else:
                st.warning(f"Файл сохранён, API ответило {r.status_code}.")
        except requests.RequestException as e:
            st.warning(f"RAG API недоступен ({e}). Настройки записаны в файл.")


def render_rag_index_section():
    s = st.session_state.rag_settings
    current_root = s.corpus_root if s.corpus_root else Path.cwd() / "outputs"
    api_url = get_rag_api_base_url()

    corpus_root = st.text_input("Папка для индексации", value=str(current_root), key="index_corpus")
    if st.button("Начать индексацию", type="primary", key="start_index"):
        try:
            resp = requests.post(
                f"{api_url}/v1/index", json={"corpus_root": corpus_root, "wait": False}, timeout=30
            )
            if resp.ok:
                st.success("Индексация запущена.")
            else:
                st.error(f"Ошибка: {resp.text}")
        except Exception as e:
            st.error(f"Не удалось подключиться к RAG: {e}")

    if st.button("Обновить статус", key="index_status"):
        try:
            resp = requests.get(f"{api_url}/v1/status", timeout=10)
            if resp.status_code == 200:
                status = resp.json().get("index_job", {})
                if status.get("phase") == "running":
                    st.info(
                        f"Индексация: {status.get('files_done')}/{status.get('files_total')} файлов, "
                        f"чанков: {status.get('chunks_upserted')}"
                    )
                elif status.get("job_id"):
                    st.success(
                        f"Последняя: {status.get('phase')}, "
                        f"{status.get('files_done')}/{status.get('files_total')} файлов"
                    )
                else:
                    st.write("Нет данных об индексации.")
        except Exception as e:
            st.error(f"Ошибка: {e}")


def render_ocr_section():
    col_input, col_settings = st.columns([1, 1])

    with col_input:
        files_input = st.text_area(
            "Пути к файлам или папкам (по одному на строку)",
            key="ocr_files",
        )
        uploaded_files = st.file_uploader("Или загрузите файлы", accept_multiple_files=True)
        out_dir = st.text_input(
            "Папка результатов",
            value=get_gui_setting("output_dir", str(Path.cwd() / "outputs")),
        )

    with col_settings:
        use_plugins = st.checkbox(
            "Плагины MarkItDown",
            value=get_gui_setting("use_plugins", True),
        )
        use_llm = st.checkbox(
            "Обработка через LLM",
            value=get_gui_setting("use_llm", False),
        )

        if use_llm:
            base_url = st.text_input("Base URL", value=get_gui_setting("base_url", ""))
            api_key = st.text_input("API-ключ", type="password", value=get_gui_setting("api_key", ""))
            mq_l, mq_r = st.columns(2)
            with mq_l:
                llm_ocr = st.text_input(
                    "OCR (vision)",
                    value=get_gui_setting("llm_ocr_model_id", get_gui_setting("ocr_model_name", "")),
                )
                llm_text = st.text_input(
                    "Текст",
                    value=get_gui_setting("llm_text_model_id", get_gui_setting("model_name", "gpt-4o")),
                )
            with mq_r:
                llm_figure = st.text_input(
                    "Рисунки (vision)",
                    value=get_gui_setting("llm_figure_model_id", ""),
                )
            use_lmstudio_autoload = st.checkbox(
                "Автозагрузка моделей в LM Studio",
                value=get_gui_setting("use_lmstudio_autoload", True),
                help="Перед OCR/текст/рисунки подгружает нужную модель. Пустой URL ниже — встроенный вызов.",
            )
            lmstudio_autoload_url = st.text_input(
                "URL autoload-сервиса (опционально)",
                value=get_gui_setting("lmstudio_autoload_url", ""),
                disabled=not use_lmstudio_autoload,
                placeholder="http://127.0.0.1:8790",
            )
        else:
            base_url = get_gui_setting("base_url", "")
            api_key = get_gui_setting("api_key", "")
            llm_ocr = get_gui_setting("llm_ocr_model_id", get_gui_setting("ocr_model_name", ""))
            llm_figure = get_gui_setting("llm_figure_model_id", "")
            llm_text = get_gui_setting("llm_text_model_id", get_gui_setting("model_name", "gpt-4o"))
            use_lmstudio_autoload = get_gui_setting("use_lmstudio_autoload", True)
            lmstudio_autoload_url = get_gui_setting("lmstudio_autoload_url", "")

        stage_options = {item["id"]: item for item in PIPELINE_STAGE_DESCRIPTIONS}
        saved_stages = get_gui_setting("enabled_stages", default_enabled_stages())
        if not isinstance(saved_stages, list):
            saved_stages = default_enabled_stages()

        st.markdown("**Системные настройки**")
        figures_workers = st.number_input(
            "Потоки для картинок", 1, 16, value=get_gui_setting("figures_workers", 4)
        )
        output_stem = st.text_input(
            "Префикс имени файла", value=get_gui_setting("output_stem", "")
        )
        enabled_stages = st.multiselect(
            "Этапы пайплайна",
            options=list(stage_options),
            default=[sid for sid in saved_stages if sid in stage_options] or default_enabled_stages(),
            format_func=lambda sid: stage_options[sid]["title"],
        )

    ocr_settings_blob = {
        "output_dir": out_dir,
        "use_plugins": use_plugins,
        "use_llm": use_llm,
        "figures_workers": figures_workers,
        "output_stem": output_stem,
        "enabled_stages": enabled_stages,
    }
    if use_llm:
        ocr_settings_blob.update(
            {
                "model_name": llm_text,
                "base_url": base_url,
                "api_key": api_key,
                "llm_ocr_model_id": llm_ocr,
                "llm_figure_model_id": llm_figure,
                "llm_text_model_id": llm_text,
                "ocr_model_name": llm_ocr,
                "use_lmstudio_autoload": use_lmstudio_autoload,
                "lmstudio_autoload_url": lmstudio_autoload_url,
            }
        )

    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        run_ocr = st.button("Запустить OCR", type="primary", use_container_width=True)
    with c2:
        save_ocr = st.button("Сохранить настройки", use_container_width=True)
    with c3:
        pause_label = "Продолжить" if st.session_state.pause_event.is_set() else "Пауза"
        if st.button(pause_label, use_container_width=True):
            if st.session_state.pause_event.is_set():
                st.session_state.pause_event.clear()
            else:
                st.session_state.pause_event.set()
            st.rerun()
    with c4:
        if st.button("Отменить", use_container_width=True):
            st.session_state.cancel_event.set()
            st.session_state.pause_event.clear()
            st.rerun()

    if save_ocr:
        apply_ocr_gui_settings_blob(ocr_settings_blob)
        st.success("Настройки сохранены.")
        st.rerun()

    if run_ocr:
        if st.session_state.job_thread and st.session_state.job_thread.is_alive():
            st.warning("Задача уже выполняется!")
        else:
            apply_ocr_gui_settings_blob(ocr_settings_blob)
            final_files = []
            if files_input.strip():
                for line in files_input.strip().split("\n"):
                    p = Path(line.strip())
                    if p.exists():
                        final_files.append(str(p))
            if uploaded_files:
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                for uf in uploaded_files:
                    p = upload_dir / uf.name
                    p.write_bytes(uf.getbuffer())
                    final_files.append(str(p))
            if not final_files:
                st.error("Укажите файлы для обработки.")
            else:
                args = {
                    "files": final_files,
                    "out_dir": out_dir,
                    "use_plugins": use_plugins,
                    "use_llm": use_llm,
                    "model_name": llm_text,
                    "base_url": base_url,
                    "api_key": api_key,
                    "llm_ocr_model_id": llm_ocr,
                    "llm_figure_model_id": llm_figure,
                    "llm_text_model_id": llm_text,
                    "use_lmstudio_autoload": use_llm and use_lmstudio_autoload,
                    "lmstudio_autoload_url": lmstudio_autoload_url if use_llm else "",
                    "ocr_model_name": llm_ocr,
                    "ocr_base_url": "",
                    "ocr_api_key": "",
                    "pdf_pages_spec": "",
                    "cancel_event": st.session_state.cancel_event,
                    "pause_event": st.session_state.pause_event,
                    "enabled_stages": enabled_stages,
                    "do_split": False,
                    "split_level": 2,
                    "keep_combined": True,
                    "split_llm_toc": False,
                    "obsidian_links": False,
                    "extract_pdf_images": True,
                    "describe_figures_llm": False,
                    "formulas_llm_latex": False,
                    "output_stem": output_stem,
                    "prefer_pdf_metadata_title": True,
                    "figures_workers": figures_workers,
                }
                st.session_state.cancel_event.clear()
                st.session_state.pause_event.clear()
                st.session_state.job_log = []
                st.session_state.job_stage_events = []
                st.session_state.job_progress = 0
                st.session_state.job_progress_max = 1
                st.session_state.job_status = "Запуск..."
                st.session_state.job_done_message = ""
                while not st.session_state.job_events.empty():
                    try:
                        st.session_state.job_events.get_nowait()
                    except queue.Empty:
                        break
                st.session_state.job_thread = threading.Thread(
                    target=run_conversion_job,
                    args=(args, st.session_state.job_events),
                    daemon=True,
                )
                st.session_state.job_thread.start()
                st.rerun()

    st.markdown("**Логи и прогресс**")
    _render_job_progress_section()


def render_rag_chat_section():
    query = st.text_area("Ваш вопрос:", key="rag_query")
    if st.button("Отправить", type="primary", key="rag_send"):
        if not query.strip():
            st.warning("Введите вопрос!")
            return
        api_url = get_rag_api_base_url()
        rs = st.session_state.rag_settings
        lm_model = (
            (getattr(rs, "lm_studio_rag_model", None) or "").strip()
            or (getattr(rs, "semantic_chunk_model", None) or "").strip()
        )
        payload: dict = {"query": query.strip()}
        if lm_model:
            payload["model"] = lm_model
        try:
            with st.spinner("Поиск и генерация ответа..."):
                resp = requests.post(f"{api_url}/v1/rag", json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer") or data.get("message", {}).get("content", "")
                    st.markdown("**Ответ**")
                    st.write(answer)
                    sources = data.get("context") or data.get("sources") or []
                    if sources:
                        st.markdown("**Источники**")
                        for idx, ctx in enumerate(sources):
                            source_file = (
                                ctx.get("source_path")
                                or ctx.get("metadata", {}).get("source_file")
                                or "?"
                            )
                            with st.container(border=True):
                                st.markdown(f"**{idx + 1}. {source_file}**")
                                st.write(ctx.get("text") or ctx.get("heading") or "")
                else:
                    st.error(f"Ошибка: {resp.text}")
        except Exception as e:
            st.error(f"RAG API недоступен: {e}")


def render_setup_doctor_section():
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Doctor", type="primary", use_container_width=True):
            _run_short_cli(["doctor"], timeout_s=60.0)
            st.rerun()
    with c2:
        if st.button("Info", use_container_width=True):
            _run_short_cli(["info"], timeout_s=30.0)
            st.rerun()
    _render_last_cli_result()

    st.subheader("Установка зависимостей")
    st.warning("Может занять долго и изменит Python-окружение.")
    available = ["rag", "web", "lmstudio", "mcp", "dev", "all"]
    selected = st.multiselect("Группы", options=available, default=["all"])
    if st.button("python llmmd.py setup ..."):
        _run_short_cli(["setup", *selected], timeout_s=1800.0)
        st.rerun()

    st.subheader("MCP для Cursor / Claude")
    st.caption("Генерирует JSON с `rag_search` и `rag_ask`. Нужен запущенный RAG API.")
    if st.button("mcp-config", key="mcp_config_btn"):
        _run_short_cli(["mcp-config"], timeout_s=30.0)
        st.rerun()
    _render_last_cli_result()


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 300;
        background-color: #fafbfc !important;
        color: #2d3748 !important;
    }

    /* Анимация появления основного контейнера */
    .block-container {
        animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Стилизация expander (сворачивающихся панелей) */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 12px !important;
        border: 1px solid #edf2f7 !important;
        padding: 12px 18px !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .streamlit-expanderHeader:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transform: translateY(-1px);
        border-color: #e2e8f0 !important;
        color: #0072ff !important;
    }
    .streamlit-expanderContent {
        border: 1px solid #edf2f7 !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        background-color: #ffffff !important;
        padding-top: 15px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }

    /* Кнопки */
    button[kind="primary"] {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 4px 10px rgba(0, 114, 255, 0.25) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(0, 114, 255, 0.35) !important;
        opacity: 0.98;
    }

    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        color: #4a5568 !important;
        font-weight: 400 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
    }
    button[kind="secondary"]:hover {
        background-color: #f7fafc !important;
        border-color: #cbd5e0 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06) !important;
        color: #2d3748 !important;
    }

    /* Поля ввода */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea, .stSelectbox > div > div > div {
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #2d3748 !important;
        transition: all 0.3s ease !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.01);
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus, .stSelectbox > div > div > div:focus {
        border-color: #0072ff !important;
        box-shadow: 0 0 0 3px rgba(0, 114, 255, 0.15) !important;
    }

    /* Заголовки */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #1a202c !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
    }

    /* Метрики (Metrics) */
    [data-testid="stMetricValue"] {
        color: #0072ff !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #718096 !important;
    }
    [data-testid="metric-container"] {
        background: #ffffff;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #edf2f7;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.08);
        border-color: #e2e8f0;
    }

    /* Прогресс-бары */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%) !important;
        border-radius: 10px !important;
    }

    /* Инфо-боксы (info, success, warning, error) */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 3px 8px rgba(0,0,0,0.04) !important;
        transition: transform 0.3s ease;
    }
    .stAlert:hover {
        transform: translateY(-1px);
    }

    /* Скроллбар */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #cbd5e0;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a0aec0;
    }
    
    /* Code blocks */
    code {
        color: #0072ff !important;
        background-color: #ebf8ff !important;
        border-radius: 6px !important;
        padding: 2px 6px !important;
    }
    pre code {
        color: inherit !important;
        background-color: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="LLMMD Lite", layout="wide", page_icon="✨")
    inject_custom_css()
    st.title("✨ LLMMD Lite")
    st.caption(f"llmmd {PROJECT_VERSION} — стильный, быстрый и минималистичный интерфейс")

    with st.expander("📊 Нагрузка на систему", expanded=True):
        render_system_section()

    with st.expander("🛠️ Сервисы (Qdrant, RAG)", expanded=False):
        render_services_section()

    with st.expander("⚙️ Настройки RAG", expanded=False):
        render_rag_config_section()

    with st.expander("🗂️ Индексация в БД", expanded=False):
        render_rag_index_section()

    with st.expander("📄 Обработка документов", expanded=False):
        render_ocr_section()

    with st.expander("💬 Чат с документами", expanded=True):
        render_rag_chat_section()

    with st.expander("🩺 Setup / Doctor", expanded=False):
        render_setup_doctor_section()


if __name__ == "__main__":
    main()
