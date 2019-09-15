/*
 *  
 * 
 * 
 */

package com.packt.scalada.viz.spark


import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.SparkSession 

/**
 * This app is just a playing ground before the fnGroupAge actually gets moved into a paragraph in Zeppelin.
 *
 */
object ZeppelinUDF extends App {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def ageGroup(age: Long) = {
    val buckets = Array("0-10", "11-20", "20-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100", ">100")
    buckets(math.min((age.toInt - 1) / 10, buckets.length - 1))
  }

  
  val spark = SparkSession.builder().config("spark.master","local").appName("Chapter 3 Scala cookbook").getOrCreate()
	 
  val profileDF = spark.read.json("profiles.json")
  

  profileDF.printSchema()

  profileDF.show()

  profileDF.createOrReplaceTempView("profiles")

  spark.udf.register("ageGroup", (age: Long) => ageGroup(age.toInt))

  spark.sql("""select ageGroup(age) as group,  count(1) as total 
                            from profiles where gender='${gender=male,male|female}' group by ageGroup(age) order by group""").show()
  spark.sql("""select eyeColor, count(eyeColor) as count from profiles where
    gender='male' group by eyeColor""").show()
                            
  
}