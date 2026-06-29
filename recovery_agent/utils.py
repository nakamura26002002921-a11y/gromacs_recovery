# recovery_agent/utils.py
import multiprocessing

def run_with_timeout(func, args, kwargs, timeout_sec):
    """
    関数を別プロセスで実行し、タイムアウトしたら強制終了する。
    PDBFixerのような重いPython処理に対して安全装置として使用する。
    """
    def wrapper(q, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            q.put(("ok", result))
        except Exception as e:
            q.put(("error", str(e)))

    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=wrapper, args=(q, *args), kwargs=kwargs)
    p.start()
    p.join(timeout_sec)

    # タイムアウトした場合
    if p.is_alive():
        p.terminate()
        p.join()
        return {
            "op_name": func.__name__, 
            "new_pdb_path": None, 
            "extra_flags": None,
            "status": "repair_timeout",
            "error": f"Repair function timed out after {timeout_sec} seconds"
        }

    # プロセスは終わったが結果がキューに入っていない場合
    if q.empty():
        return {
            "op_name": func.__name__, 
            "new_pdb_path": None, 
            "extra_flags": None,
            "status": "repair_error",
            "error": "Process finished without returning a result"
        }

    status, result = q.get()
    if status == "error":
        raise RuntimeError(result)
    return result
