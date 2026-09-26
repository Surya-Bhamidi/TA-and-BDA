"""Actual Spark SQL cleaning, SQL aggregates, Parquet, and MLlib classification."""
import time
import os
import socket
from pathlib import Path

from .config import RAW, PROCESSED, ARTIFACTS, ROOT, SEED, write_json


def create_spark(master="local[2]"):
    from pyspark.sql import SparkSession
    builder = (SparkSession.builder.master(master).appName("DecodingCrimeNarratives")
               .config("spark.sql.shuffle.partitions", "8")
               .config("spark.driver.memory", "2g")
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.ansi.enabled", "false")
               .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
               .config("spark.eventLog.enabled", "false"))
    if master.startswith("local"):
        builder = builder.config("spark.driver.bindAddress", "127.0.0.1")
    else:
        os.environ["SPARK_LOCAL_IP"] = socket.gethostbyname(socket.gethostname())
        builder = (builder.config("spark.driver.bindAddress", "0.0.0.0")
                   .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", socket.gethostname())))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def clean_frame(raw):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    required = ["report_id", "narrative", "crime_type", "district", "reported_at", "split", "template_group", "source", "entities_json"]
    from .config import CATEGORIES
    valid = raw.dropna(subset=required).filter(F.length(F.trim(F.col("narrative"))) > 0)
    valid = valid.withColumn("event_time", F.to_timestamp("reported_at"))
    valid = valid.filter(F.col("event_time").isNotNull() & F.col("crime_type").isin(CATEGORIES)
                         & F.col("split").isin("train", "validation", "test"))
    # Keep original text unchanged: NER offsets refer to it. Clean text is a second column.
    valid = valid.withColumn("clean_text", F.trim(F.regexp_replace(F.lower("narrative"), r"[^\p{L}\p{N}\s]", " ")))
    valid = valid.withColumn("clean_text", F.trim(F.regexp_replace("clean_text", r"\s+", " ")))
    valid = valid.filter(F.length("clean_text") > 0)
    valid = valid.withColumn("text_hash", F.sha2("clean_text", 256))
    # Deterministic canonical record (row_number avoids nondeterministic dropDuplicates survivors).
    canonical = valid.withColumn("_rank", F.row_number().over(Window.partitionBy("text_hash").orderBy("report_id")))
    clean = canonical.filter("_rank = 1").drop("_rank")
    clean = clean.withColumn("month", F.date_format("event_time", "yyyy-MM"))
    return valid, clean


def run_spark(master="local[2]", raw_path=RAW):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructType, StructField, StringType
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    started = time.perf_counter()
    spark = create_spark(master)
    try:
        schema = StructType([StructField(k, StringType(), True) for k in ["report_id", "narrative", "crime_type", "reported_at", "district", "template_group", "split", "source", "entities_json"]])
        raw = spark.read.schema(schema).json(str(Path(raw_path).resolve())).repartition(8).cache()
        raw_count = raw.count()
        valid, clean = clean_frame(raw)
        valid_count = valid.count()
        clean = clean.cache()
        clean_count = clean.count()
        clean.write.mode("overwrite").partitionBy("split").parquet(str(PROCESSED / "reports"))
        clean.createOrReplaceTempView("crime_reports")
        spark.sql("SELECT month, crime_type, district, count(*) AS reports FROM crime_reports GROUP BY month, crime_type, district ORDER BY month, crime_type, district").toPandas().to_parquet(PROCESSED / "trends.parquet", index=False)
        # Independent distributed MLlib baseline, using train/test scenario partitions.
        labeler = StringIndexer(inputCol="crime_type", outputCol="label", stringOrderType="alphabetAsc")
        pipeline = Pipeline(stages=[labeler, RegexTokenizer(inputCol="clean_text", outputCol="words", pattern=r"\W+"),
            StopWordsRemover(inputCol="words", outputCol="filtered"),
            HashingTF(inputCol="filtered", outputCol="tf", numFeatures=4096),
            IDF(inputCol="tf", outputCol="features"),
            LogisticRegression(maxIter=25, regParam=.05, elasticNetParam=0., featuresCol="features", labelCol="label")])
        model = pipeline.fit(clean.filter("split = 'train'"))
        predictions = model.transform(clean.filter("split = 'test'")).cache()
        metrics = {name: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=name).evaluate(predictions) for name in ["accuracy", "f1"]}
        model.write().overwrite().save(str(ARTIFACTS / "spark_mllib_model"))
        report = {"engine": "Apache Spark", "spark_version": spark.version, "master": spark.sparkContext.master,
                  "input_partitions": raw.rdd.getNumPartitions(), "shuffle_partitions": 8,
                  "raw_rows": raw_count, "invalid_rows_removed": raw_count - valid_count,
                  "duplicates_removed": valid_count - clean_count, "clean_rows": clean_count,
                  "storage": "Spark-written Parquet partitioned by split", "runtime_seconds": round(time.perf_counter() - started, 2),
                  "mllib": {"model": "HashingTF + IDF + multinomial logistic regression", "train_rows": clean.filter("split = 'train'").count(), "test_rows": predictions.count(), "accuracy": metrics["accuracy"], "weighted_f1": metrics["f1"]},
                  "deployment_note": "local[2] executes Spark tasks on two local cores; this run is not a multi-machine cluster."}
        write_json(ARTIFACTS / "spark_metrics.json", report)
        print(f"Spark cleaned {clean_count:,} reports; held-out MLlib accuracy={metrics['accuracy']:.4f}", flush=True)
        return report
    finally:
        spark.stop()
