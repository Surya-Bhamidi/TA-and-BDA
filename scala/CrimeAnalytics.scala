// Scala 2.12 / Spark 3.5. Supplements the functional-programming syllabus.
// Compile and run with scripts/run_scala.py; Python remains the main NLP pipeline.
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions._

object CrimeAnalytics {
  case class CaseRecord(report_id: String, narrative: String, crime_type: String)
  trait TextTransform { def apply(text: String): String }
  object Normalize extends TextTransform {
    override def apply(text: String): String = text.toLowerCase(java.util.Locale.ROOT).replaceAll("\\s+", " ").trim
  }
  def categoryFamily(category: String): String = category match {
    case "Cybercrime" | "Fraud" => "Digital / financial"
    case "Assault" | "Robbery" => "Interpersonal"
    case _ => "Property"
  }
  def main(args: Array[String]): Unit = {
    require(args.length == 1, "Pass the Spark reports Parquet directory")
    val spark = SparkSession.builder.appName("CrimeNarrativesScala").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    import spark.implicits._
    try {
      val reports: Dataset[CaseRecord] = spark.read.parquet(args(0))
        .select("report_id", "narrative", "crime_type").as[CaseRecord]
      // Immutable case classes, higher-order functions, closures and typed collections.
      val minimumLength = 1 // Short informal reports are valid too.
      val cleaned = reports.filter(_.narrative.trim.length >= minimumLength)
        .map(r => r.copy(narrative = Normalize(r.narrative)))
      val familyCounts = cleaned.map(r => categoryFamily(r.crime_type))
        .groupBy("value").count().orderBy(desc("count"))
      val stopwords = Set("the", "a", "an", "and", "of", "to", "was", "in")
      val wordCounts = cleaned.flatMap(_.narrative.split("[^\\p{L}\\p{N}]+").toSeq)
        .filter(w => w.length > 2 && !stopwords.contains(w))
        .groupBy("value").count().orderBy(desc("count"))
      println(s"SCALA_VALID_REPORTS=${cleaned.count()}")
      familyCounts.show(false)
      wordCounts.show(15, false)
      cleaned.createOrReplaceTempView("scala_cases")
      spark.sql("SELECT crime_type, count(*) AS cases FROM scala_cases GROUP BY crime_type ORDER BY cases DESC").show(false)
      val example: List[String] = List("a report", "another report")
      val indexed: Map[Int, String] = example.zipWithIndex.map { case (text, i) => i -> Normalize(text) }.toMap
      println(s"Immutable collection example: $indexed")
    } finally { spark.stop() }
  }
}
