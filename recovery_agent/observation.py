# recovery_agent/observation.py
import subprocess
import shutil
import os

class ObservationModule:
    def __init__(self, force_field, water_model):
        self.force_field = force_field
        self.water_model = water_model
        self._check_gmx_installed()

    def _check_gmx_installed(self):
        if not shutil.which("gmx"):
            raise EnvironmentError("GROMACS ('gmx' command) is not found in PATH.")

    def run_pdb2gmx(self, pdb_path, work_dir, additional_flags=None, timeout=120):
        """
        pdb2gmxを実行する。
        work_dir: 作業ディレクトリ。出力ファイルはここに生成される。
        timeout: タイムアウト秒数。デフォルト120秒。
        """
        if additional_flags is None:
            additional_flags = []

        # 入力ファイルは絶対パスで指定し、作業ディレクトリ外にあっても参照できるようにする
        abs_pdb_path = os.path.abspath(pdb_path)

        cmd = [
            "gmx", "pdb2gmx",
            "-f", abs_pdb_path,
            "-o", "temp_conf.gro",
            "-p", "temp_topol.top",
            "-ff", self.force_field,
            "-water", self.water_model
        ] + additional_flags

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=work_dir,             # ★ケースごとの作業ディレクトリ内で実行
                stdin=subprocess.DEVNULL, # ★対話プロンプトでハングしないよう即座にEOF
                timeout=timeout           # ★タイムアウト設定
            )
            success = result.returncode == 0
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"TIMEOUT: gmx pdb2gmx did not finish in {timeout} seconds",
                "returncode": -1
            }
            
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
