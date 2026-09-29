"""Use the Scala compiler shipped with PySpark; no additional Scala install required."""
import os
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, ARTIFACTS, PROCESSED, write_json


def main():
    import pyspark
    from pyspark.find_spark_home import _find_spark_home
    spark_home = Path(_find_spark_home())
    java = Path(os.environ["JAVA_HOME"]) / "bin" / ("java.exe" if os.name == "nt" else "java")
    classes = ARTIFACTS / "scala_classes"
    classes.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(java), "-cp", str(spark_home / "jars" / "*"), "scala.tools.nsc.Main", "-usejavacp", "-d", str(classes), str(ROOT / "scala" / "CrimeAnalytics.scala")], check=True)
    jar = ARTIFACTS / "crime-analytics-scala.jar"
    with zipfile.ZipFile(jar, "w", zipfile.ZIP_DEFLATED) as out:
        for file in classes.rglob("*.class"):
            out.write(file, file.relative_to(classes).as_posix())
    submit = spark_home / "bin" / ("spark-submit.cmd" if os.name == "nt" else "spark-submit")
    result = subprocess.run([str(submit), "--master", "local[2]", "--class", "CrimeAnalytics", str(jar), str(PROCESSED / "reports")],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    (ROOT / "logs" / "scala.log").write_text(result.stdout, encoding="utf-8")
    match = re.search(r"SCALA_VALID_REPORTS=(\d+)", result.stdout)
    actual = int(match[1]) if match else None
    expected = json.loads((ARTIFACTS / "spark_metrics.json").read_text(encoding="utf-8"))["clean_rows"]
    verified = result.returncode == 0 and actual == expected
    write_json(ARTIFACTS / "scala_verification.json", {"exit_code": result.returncode, "records_verified": verified,
               "rows": actual, "expected_rows": expected, "temporary_jar_cleanup_warning": "Exception while deleting Spark temp dir" in result.stdout})
    if not verified:
        raise RuntimeError(f"Scala verification failed: {actual} reports, expected {expected}. See logs/scala.log.")
    print(f"SCALA_VALID_REPORTS={actual}; verified against the current Spark corpus.")


if __name__ == "__main__":
    main()
