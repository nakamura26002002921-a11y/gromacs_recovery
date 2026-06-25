import subprocess
import shutil

class ObservationModule:
    def __init__(self, force_field, water_model):
        self.force_field = force_field
        self.water_model = water_model
        self._check_gmx_installed()

    def _check_gmx_installed(self):
        if not shutil.which("gmx"):
            raise EnvironmentError("GROMACS ('gmx' command) is not found in PATH.")

    def run_pdb2gmx(self, pdb_path, additional_flags=None):
        if additional_flags is None:
            additional_flags = []
            
        cmd = [
            "gmx", "pdb2gmx",
            "-f", pdb_path,
            "-o", "temp_conf.gro",
            "-p", "temp_topol.top",
            "-ff", self.force_field,
            "-water", self.water_model
        ] + additional_flags

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            success = result.returncode == 0
            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}
