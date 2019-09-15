package com.packt.scalada.viz.breeze
import ML_scalaAdvanceMethods._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}

import org.apache.log4j._

 
 //vectorizing
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.{ KMeans=> KMeansML}

object BigDataMartSalesML_LiRgrDF extends App {
        Logger.getLogger("org").setLevel(Level.ERROR)
        val spark = SparkSession
                    .builder()
                    .appName("Java Spark SQL basic example")
                    .config("spark.master", "local")
                    .getOrCreate()
        import spark.sqlContext.implicits._    
        
        var rawDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").
                     option("inferSchema","true").load("/Users/keeratjohar2305/Downloads/Dataset/AV_trainBigDataMartSales.txt")        

        var rawTestDF = spark.read.format("csv").option("Delimiter", ",").option("header","true").
                     option("inferSchema","true").load("/Users/keeratjohar2305/Downloads/Dataset/AV_testBigDataMartSales.txt")                     
        
        ML_scalaAdvanceMethods.dsShape(rawDF)
        ML_scalaAdvanceMethods.dsShape(rawTestDF)
        rawDF.show(20)
        rawDF.columns
        rawDF.printSchema()
        ML_scalaAdvanceMethods.summaryCustomized(rawDF).show()
        ML_scalaAdvanceMethods.summaryCustomized(rawTestDF).show()
        

 
// Print Data from categrical columns
println ("\nFrequency of Categories for categorical variables:-")
rawDF.columns.foreach{ fieldName =>
                           if ( Seq("Item_Fat_Content","Item_Type","Outlet_Identifier","Outlet_Establishment_Year","Outlet_Size","Outlet_Location_Type","Outlet_Type").
                                            contains(fieldName)
                              )
                                 println(fieldName,":",rawDF.groupBy(fieldName).count().collect().toList)
                          else
                                 None
                      }

        System.exit(1)
//1) missing value Fill for numeric continues column
val rawDFMissingFilled  = missingValFilled(rawDF,"Item_Weight")  
 // rawDFCleaned.na.replace("Outlet_Size", )
  //summaryCustomized(rawDFMissingFilled).show()

 //2) Missing values filling based one column based on another
val myFunct = udf((str: String)=> if (str!=null && Array("Supermarket Type1",  "Supermarket Type3" ,"Supermarket Type2" ,"Grocery Store").contains(str) ) 
                                  Map("Supermarket Type1"-> "High", "Supermarket Type3"-> "small" ,  "Supermarket Type2"-> "small","Grocery Store"-> "Mediaum")(str)
                                  else ""
                  )             
spark.udf.register("myFunct", myFunct)

val NewDF= rawDFMissingFilled.withColumn("Outlet_Size1",when (col("Outlet_Size").isNull, myFunct(col("Outlet_Type"))).otherwise(col("Outlet_Size")))
//val mapOutlet_Type = Map("Supermarket Type1"-> "High", "Supermarket Type3"-> "small" ->  "Supermarket Type2"-> "small","Grocery Store,"-> "Mediaum")
    
summaryCustomized(NewDF).show()



val Array(traningDF,testingDF) = NewDF.randomSplit(Array(0.7,0.3),seed=99999999)




println("\n***********   END: sales Data Exploration and missing Values Replacement *****************\n")

println("\n***********   Start: sales Data ML piple Creation *****************\n")





// Seperate out String feature column and numeric features
 val featuresCatColNames = Seq("Item_Identifier", "Item_Fat_Content", "Item_Type","Outlet_Identifier","Outlet_Size1","Outlet_Location_Type","Outlet_Type")
 val featuresNumColNames = Seq("Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Establishment_Year","Item_Outlet_Sales")
 

 val stages = CategoricalFeatureVectorzing(featuresCatColNames) ++ FeatureAssembler(featuresCatColNames,featuresNumColNames )
  
// Setup k-means model with two clusters
val pipelineInstatiated = new Pipeline().setStages(stages)

val pipedtrainingDF = pipelineInstatiated.fit(traningDF).transform(traningDF)
println("pipedDF")
pipedtrainingDF.show(false)


println("\n***********   End: sales Data ML piple Creation *****************\n")



println("trained DF")

 //Lets train K means model
val MLkmeans = new KMeansML().setK(10).setSeed(1L)
val MLkmeansTrainedDF = MLkmeans.fit(pipedtrainingDF)
val Kmean_prediction =  MLkmeansTrainedDF.transform(pipedtrainingDF)

 //Kmean_prediction.select(col("prediction")).distinct().collect().map(_.mkString.toInt)
 import spark.sqlContext.implicits._
 
 (0 to 9).foreach(x=>
 Kmean_prediction.filter(col("prediction")===x).show(4)
 )
 


//lets make the pridiction kmeansPipeline
// create piped test DF
 //val pipedtestDF = pipelineInstatiated.fit(testingDF).transform(testingDF)
 //println("pipedtestDF testing DF")
 //pipedtestDF.show(false)
 
            
}