
__all__ = [
    "_is_running_with_streamlit",
]

def _is_running_with_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False