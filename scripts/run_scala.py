"""Use the Scala compiler shipped with PySpark; no additional Scala install required."""
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crime_nlp.config import ROOT, ARTIFACTS, PROCESSED


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
    subprocess.run([str(submit), "--master", "local[2]", "--class", "CrimeAnalytics", str(jar), str(PROCESSED / "reports")], check=True)


if __name__ == "__main__":
    main()
