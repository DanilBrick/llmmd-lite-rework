"""Форматирование длительности для логов и строки статуса (рус.)."""


def format_duration_ru(seconds: float) -> str:
    if seconds != seconds or seconds < 0:  # NaN / invalid
        return "—"
    s = int(round(seconds))
    if s < 60:
        return f"{s} с"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} мин {sec} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m} мин {sec} с"
