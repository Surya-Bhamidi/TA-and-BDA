"""Start a real standalone Spark master + two workers, run ETL, and stop owned processes.

All three processes run on this computer. This demonstrates worker/executor
separation, not performance across physical machines. No Docker is required.
"""
import json
import os
import socket
import subprocess
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, ARTIFACTS, write_json


def main():
    for port in (17077, 18080, 18081, 18082, 17101, 17102):
        with socket.socket() as check:
            try:
                check.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(f"Port {port} is busy; no existing process was stopped.") from exc
    spark_home = Path(os.environ["SPARK_HOME"])
    launcher = spark_home / "bin" / ("spark-class.cmd" if os.name == "nt" else "spark-class")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    children, logs = [], []
    def start(name, args):
        log = (ROOT / "logs" / f"cluster-{name}.log").open("w", encoding="utf-8")
        logs.append(log)
        # Spark's Windows batch launcher can reuse the same %RANDOM% filename
        # when several JVMs start together. Give each launcher its own temp area.
        launch_temp = tempfile.mkdtemp(prefix=f"crime-spark-{name}-")
        environment = dict(os.environ, TEMP=launch_temp, TMP=launch_temp)
        child = subprocess.Popen([str(launcher), *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags, env=environment)
        children.append(child)
    try:
        start("master", ["org.apache.spark.deploy.master.Master", "--host", "127.0.0.1", "--port", "17077", "--webui-port", "18080"])
        for number in (1, 2):
            start(f"worker{number}", ["org.apache.spark.deploy.worker.Worker", "spark://127.0.0.1:17077", "--host", "127.0.0.1", "--port", str(17100 + number), "--webui-port", str(18080 + number), "--cores", "1", "--memory", "1g", "--work-dir", str(ROOT / ".runtime" / f"worker{number}")])
        state = None
        for _ in range(60):
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/json/", timeout=2) as response:
                    state = json.load(response)
                if len([w for w in state.get("workers", []) if w.get("state") == "ALIVE"]) == 2:
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Two Spark workers did not register. Inspect logs/cluster-*.log.")
        print("Two standalone Spark workers registered; running ETL and MLlib.", flush=True)
        subprocess.run([sys.executable, "run_pipeline.py", "--stage", "spark", "--master", "spark://127.0.0.1:17077"], cwd=ROOT, check=True)
        write_json(ARTIFACTS / "cluster_execution.json", {"status": "completed", "master": "spark://127.0.0.1:17077", "workers": state["workers"],
                   "scope": "One physical Windows host; one master, two independent worker JVMs, two execution cores. Not a multi-host benchmark.", "cleanup": "owned process trees terminated after completion"})
    finally:
        for child in reversed(children):
            if child.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
