package com.packt.scalada.viz.breeze

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark._
import org.apache.spark.sql.SQLContext
//import com.databricks.spark.csv.CsvContext
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.types.StringType

import scala.util.Try
import org.apache.spark.sql.types.DateType
import org.json4s.JObject
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.SparkSession
object DataFrameCreation_DataBricks extends App {
   Logger.getLogger("org").setLevel(Level.ERROR)
   val spark = SparkSession.builder().config("spark.master","local").appName("Chapter 3 Scala cookbook").getOrCreate()
   import spark.sqlContext.implicits._
   
   // Example1
   case class MapEntry(key: String, value: Int)
   val largeSeries = for (x <- 1 to 5000) yield MapEntry("k_%04d".format(x), x)
   
   
   val largeDataFrame = spark.sparkContext.parallelize(largeSeries).toDF()
   largeDataFrame.createOrReplaceTempView("largeTable")
   spark.sql("select * from largeTable").show()
   
   // Example 2
   case class PivotEntry(key: String, series_grouping: String, value: Int)
   val largePivotSeries = for (x <- 1 to 5000) yield 
                 PivotEntry("k_%03d".format(x % 200),  "group_%01d".format(x % 3), x)
   val largePivotDataFrame = spark.sparkContext.parallelize(largePivotSeries).toDF()
   largePivotDataFrame.createOrReplaceTempView("table_to_be_pivoted")
   spark.sql("select * from table_to_be_pivoted").show()
   
   // Example 3 
   case class SalesEntry(category: String, product: String, year: Int, salesAmount: Double)
   val salesEntryDataFrame = spark.sparkContext.parallelize(
                                              SalesEntry("fruits_and_vegetables", "apples", 2012, 100.50) :: 
                                              SalesEntry("fruits_and_vegetables", "oranges", 2012, 100.75) :: 
                                              SalesEntry("fruits_and_vegetables", "apples", 2013, 200.25) :: 
                                              SalesEntry("fruits_and_vegetables", "oranges", 2013, 300.65) :: 
                                              SalesEntry("fruits_and_vegetables", "apples", 2014, 300.65) :: 
                                              SalesEntry("fruits_and_vegetables", "oranges", 2015, 100.35) ::
                                              SalesEntry("butcher_shop", "beef", 2012, 200.50) :: 
                                              SalesEntry("butcher_shop", "chicken", 2012, 200.75) :: 
                                              SalesEntry("butcher_shop", "pork", 2013, 400.25) :: 
                                              SalesEntry("butcher_shop", "beef", 2013, 600.65) :: 
                                              SalesEntry("butcher_shop", "beef", 2014, 600.65) :: 
                                              SalesEntry("butcher_shop", "chicken", 2015, 200.35) ::
                                              SalesEntry("misc", "gum", 2012, 400.50) :: 
                                              SalesEntry("misc", "cleaning_supplies", 2012, 400.75) :: 
                                              SalesEntry("misc", "greeting_cards", 2013, 800.25) :: 
                                              SalesEntry("misc", "kitchen_utensils", 2013, 1200.65) :: 
                                              SalesEntry("misc", "cleaning_supplies", 2014, 1200.65) :: 
                                              SalesEntry("misc", "cleaning_supplies", 2015, 400.35) ::
                                              Nil).toDF()
  salesEntryDataFrame.createOrReplaceTempView("test_sales_table")
  spark.sql("select * from test_sales_table").show()
   
}