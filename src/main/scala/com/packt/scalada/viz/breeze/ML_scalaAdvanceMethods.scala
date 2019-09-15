package com.packt.scalada.viz.breeze


import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,OneHotEncoderEstimator,Normalizer}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator 


object ML_scalaAdvanceMethods  {
   
  def  getSparkSessionConnection(appName: String) : org.apache.spark.sql.SparkSession = { 
      val spark = SparkSession
              .builder()
              .appName(appName)
              .config("spark.master", "local")
              .getOrCreate()
      import spark.sqlContext.implicits._
      spark
  }
  
  def dataFitnessCheck(df: org.apache.spark.sql.DataFrame)={
    println("Checking If null values is there is dataset or not")
    println(s"		Total Reconds in Dataset is ${df.count()}   ")
    println(s"		Total Null reconds in Dataset is ${df.count() - df.na.drop().count() } ")
  }
  
  def summaryCustomized(df: org.apache.spark.sql.DataFrame): org.apache.spark.sql.DataFrame= {
     
    // If the columns data type is feature in the Dataframe m it will not work
    var paramName = "countDistinct" 
    val ColumnListWithVariancedistinct_count = Array(paramName) ++ df.columns
    val WithDistinctCntSummaryDF=   df.
                                    select(df.columns.map(c => countDistinct(col(c)).alias(c)): _*).  
                                    withColumn(paramName, lit(paramName)).
                                    selectExpr(ColumnListWithVariancedistinct_count:_*)
    
    paramName = "variance" 
    val ColumnListWithVariance = Array(paramName) ++ df.columns
    val WithVarianceSummaryDF=   df.
                                 select(df.columns.map(c => variance(col(c)).alias(c)): _*).
                                 withColumn(paramName, lit(paramName)).
                                 selectExpr(ColumnListWithVariance:_*)
                                 
    
    val ColTypeDF =  df.dtypes.map(_._2.substring(0,4))
     paramName = "NullValueCount"
     val ColumnListWithNullValueCount = Array(paramName) ++ df.columns
     val ColumnListWithNullValueCountDF=    df.
                                            select(df.columns.map(c => sum(when( col(c).isNull,  lit(1) ).otherwise(lit(0))  ).alias(c)): _*).
                                            withColumn(paramName, lit(paramName)).
                                            selectExpr(ColumnListWithNullValueCount:_*)
    
   df.summary().union(WithDistinctCntSummaryDF).union(WithVarianceSummaryDF).union(ColumnListWithNullValueCountDF)
 }
 
 def missingValFilled(DFmissingCol: org.apache.spark.sql.DataFrame,coloumnName: String) : org.apache.spark.sql.DataFrame ={
       val colDataDF:org.apache.spark.sql.DataFrame  = DFmissingCol.select(col(coloumnName)).filter(col(coloumnName).isNull)
       println("missing count for Column= " +  coloumnName +" is "+ colDataDF.count())
    
      if (colDataDF.count()> 0)
          {
          //println("I am here"+ coloumnName)
           //DFmissingCol.select(coloumnName).show()
           val mean_colData: Double = DFmissingCol.select(coloumnName).filter(col(coloumnName).isNotNull).rdd.map(x=> x.mkString.trim().toDouble).mean()//.asInstanceOf[Double]
           println("mean_colData count= "+ mean_colData)
           val fill_MissingValuesDF:org.apache.spark.sql.DataFrame  = DFmissingCol.na.fill(mean_colData,Seq(coloumnName))
         
           println("Missing rows for " + coloumnName + 
                      "\nBefore:  "+ colDataDF.count() + 
                      "\nand After: " + fill_MissingValuesDF.select(col(coloumnName)).filter(col(coloumnName).isNull).count() + 
                      "\nits Mean  =  " + mean_colData)
         
          fill_MissingValuesDF
          }
      else 
          DFmissingCol
     //fill_MissingValuesDF
      }
          
 
 
  def dsShape(df: org.apache.spark.sql.DataFrame ) ={
           val columns_cnt= df.columns.length
           val row_cnt= df.count()      
           print(s"Shape(rows =$row_cnt and colummns = $columns_cnt) \n ")
        }
  
  def univariateAnalysis( df: org.apache.spark.sql.DataFrame, CatoricalCols: Seq[String]) {
    CatoricalCols.intersect(df.columns).foreach{ colName=>
      println("Univeriate Exploratory Data Analysis-Univariate for Column : " +colName + "\n")
            println ( colName + ":   " + df.groupBy(colName).count().orderBy(col("count").asc_nulls_last).collect().toList)
    }
  }
  
  def Confusion_matrixCalculation(pridictedDF: org.apache.spark.sql.DataFrame, algoName: String)= {
    
//      println(s"""\n               ************************************************************************************************ 
//               *
//               *                      Confusion Matrix for ${algoName}
//               *          
//               ************************************************************************************************ \n""")
       val tatal = pridictedDF.count()
       val correct_prediction = pridictedDF.filter(col("label") === col("prediction")).count()
       val wrong_prediction = pridictedDF.filter(col("label") =!= col("prediction")).count()
       println(s"""\n tatal prediction =  $tatal \n correct_prediction = $correct_prediction \n wrong_prediction = $wrong_prediction  """)
       //print(pridictedDF.dtypes.toList)
       val trueP = pridictedDF.filter(col("label") === col("prediction") && col("label") === 1.0 && col("prediction") === 1.0 ).count()
       val trueN = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 1.0 && col("prediction") === 0.0).count()
       
       val falseN = pridictedDF.filter(col("label") === col("prediction") && col("label") === 0.0 && col("prediction") === 0.0 ).count()
       val falseP = pridictedDF.filter(col("label") =!= col("prediction") && col("label") === 0.0 && col("prediction") === 1.0).count()
       
       println(s" \n trueP = $trueP \n trueN = $trueN \n falseN = $falseN \n  falseP = $falseP ")
       
       val ratioWrong=wrong_prediction.toDouble/tatal.toDouble
       val ratioCorrect=correct_prediction.toDouble/tatal.toDouble
          
       println("ratioWrong =", ratioWrong)
       println("ratioCorrect =", ratioCorrect)
   }
  
  def BinaryClassificationEvaluator_ROC (pridictedDF: org.apache.spark.sql.DataFrame,algoName : String) = {
    
      // Sensitivity: probability (prediction is positive )when you have actual disease TP ( 1,1) => box(a)
       /*
        * 			True positive = correctly identified
							False positive = incorrectly identified
							True negative = correctly rejected
							False negative = incorrectly rejected
           Recall : TPR/ Sensitivity/ Hit Rate => probability of Correctly Identifying the  positive Situation
           Precision: TNR / Specificity/ Selectivity => probability of correctly assessing the negative situation
           
           FPR : probability of  incorrectly assessing the positive situations
           FNR : probability of incorrectly assessing the negative situations
        */
      // specificity: probability ( prediction is negative) when you do no have actual disease(0,0) => box(d)
//     println(s"""\n               ************************************************************************************************ 
//               *
//               *                      BinaryClassificationEvaluator_ROC Matrix for ${algoName}
//               *                      {TP = a, TN =d  } { FP = c  FN = b}
//               *                       1) 0.5 < AUC < 1  
//                                  {  its Curve of TP vs FP or a vs c  OR Sensitivity vs Specificity or TPR vs FPR 0R Y- Axis or X- Axis}
//                             
//                                       2) matrix cost calc = (-1* TP + 100* FP + 1 * TN) => -1 * a +  100* b + 1*a)
//               *                      Some cases consider model with Less cost Matrix, In some other cases with higher accurary matters
//               
//               *                       3) TPR/sensitivity/recall/hit rate =>  a / (a + c) => TP /  (TP + FN)
//               *                          TNR/specificity =>     d / ( d + b) => TN / ( TN + FP)  
//                                         
//                                          FPR/miss rate =  c /  c + d  =  FP / ( FP + TN)
//                                          FNR = b / a + b  = FN / (FN + TP)
//                                          
//                                          precision/ +vePredictiveVal = TP / (TP + FP)  
//                                                     -vePredictiveVal = TN / (TN + FN) 
//                                          
//                                       5) F - measure = 2a / (2a + b + c) = > 2TP / (2TP + TN + FP)
//                                          
//                                       7) How far Off is the predicted value form the acutal value 
//                                          Linear Regression Loss functions
//                                          
//                                          test Error : Remember =  y' - mean and y'1 = predicted values , y or x = actual value
//                                         1) Squared Error =  (x - x'1) *  (x - x'1),                                      
//                                         2) Mean Square Error =  Sum (Absolute Error)  / d             
//                                         3) Root Mean Square Error = Root Of (Mean Square Error)    
//                                         4) Relative Square Error = Mean Square Error  / Sum ((x - x') *  (x - x'))                                         
//                                         5) Root Relative Squared Error= Root  (Relative Square Error)
//                                         
//                                         A)  Absolute Error = modulas of | y - y'1| 
//                                         B)  mean Absolute Error =  Sum(Absolute Error)  / d
//                                         C)  Relative Absalute Error =  Sum(Absolute Error )/ Sum (|y - y'|)
//
//                                  
//               ************************************************************************************************ \n""")
   val evaluatorROC = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")
      
   val accuracyROC = evaluatorROC.evaluate(pridictedDF)
   println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderROC): $accuracyROC")
   println ( "lets train model")
   
 }
 
 def BinaryClassificationEvaluator_PR (pridictedDF: org.apache.spark.sql.DataFrame,algoName:String) = {
//    println(s"""\n               ************************************************************************************************ 
//               *
//               *                      BinaryClassificationEvaluator_PR Matrix for ${algoName}
//               *          
//               ************************************************************************************************ \n""")
   
   val evaluatorPR = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderPR")
   val accuracyPR = evaluatorPR.evaluate(pridictedDF)
   
   println(s"Accuracy  Per BinaryClassificationEvaluator(areaUnderPR): $accuracyPR")
     
 }
 
  def EDA_BivariateAnalysis( df: org.apache.spark.sql.DataFrame , ColumnArray: Array[(String,String)])  = {
         
           ColumnArray.foreach(x   => 
          
           println(s"""Exploratory Data Analysis - Binvariate
          
           df.stat.corr( ${x._1},${x._2}):    """  + df.stat.corr(x._1,x._2) ))
         
           //System.exit(1)
        }
  
  def CallNaiveBayesAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame, dataSetType:String )={
  
   import org.apache.spark.ml.classification.NaiveBayes
                // instantiate the base classifier
    println("Baddest for binary classfication")
    val nv = new NaiveBayes()   
    val nv_prediction = nv.fit(pipedDF).transform(pipedtestDF)//.
//    
//    drop("features").
//    drop("GenderVector").
//    drop("MarriedVector").
//    drop("DependentsVector").
//    drop("EducationVector").
//    drop("Self_EmployedVector").
//    drop("Credit_HistoryVector").
//    drop("Property_AreaVector").
//    drop("rawPrediction").
//    drop("probability")
//    nv_prediction.show(20)
//    nv_prediction.write.format("csv").option("header", "true").save("/Users/keeratjohar2305/Downloads/Dataset/load_pridicted.csv")
//    System.exit(1)
    //nv_prediction.show()
    
    println("--------------------------------------------------------------Applying General Default NaiveBayes ------------------------------------------")
     nv_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)
             
    if (dataSetType == "MultiClass")
            {
              
              MultiClassConfusionMatrix(nv_prediction)
            }
    else {       
         //Lets mearsure the accruarcy
         Confusion_matrixCalculation(nv_prediction,"General Default NaiveBayes")
         BinaryClassificationEvaluator_ROC(nv_prediction,"General Default NaiveBayes")
         BinaryClassificationEvaluator_PR(nv_prediction,"General Default NaiveBayes")
 
          }   

  }
  
   def CallMultiLayerPerceptrolAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame, dataSetType:String )={
  
   import org.apache.spark.ml.classification.NaiveBayes
                // instantiate the base classifier
   /*
    * 
    *  Attribute value at array index	Description
    *  0	-   This is the number of neurons or perceptrons at the input layer of the network. This is the count of the number of features that are
    *  passed to the model.
    *  1 - 	This is a hidden layer containing five perceptrons (sigmoid neurons only, ignore the terminology).
    *  2 - 	This is another hidden layer containing four sigmoid neurons.
    *  3 - 	This is the number of neurons representing the output label classes. In our case, we have three types of Iris flowers, hence three classes.
    */
  
       import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
       val layers = Array[Int](4, 5, 4, 3)
       val mlpc = new MultilayerPerceptronClassifier()
                        .setLayers(layers)
                        .setBlockSize(128)
                        .setSeed(1234L)
                        .setMaxIter(100)
      
       val mlpc_prediction = mlpc.fit(pipedDF).transform(pipedtestDF)
       mlpc_prediction.show()
       println("--------------------------------------------------------------Applying General Multilayer perceptron Classfier ------------------------------------------")
       println("Here expected 3 categorical values for input column label, but the input column had metadata specifying 2 values ")
       mlpc_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)
                if (dataSetType == "MultiClass")
                        {
                          
                          MultiClassConfusionMatrix(mlpc_prediction)
                        }
                else {       
                     //Lets mearsure the accruarcy
                      println("It do not support Binary Classification")
       
                }   
    
   }
  def MultiClassConfusionMatrix(ovs_prediction: org.apache.spark.sql.DataFrame)={
      // evaluate the model
          import org.apache.spark.mllib.evaluation.MulticlassMetrics
          import org.apache.spark.ml.util.MetadataUtils
          val predictionsAndLabels = ovs_prediction.select("prediction", "label").rdd.map(row => (row.getDouble(0), row.getDouble(1)))
          
          val metrics = new MulticlassMetrics(predictionsAndLabels)
          
          val confusionMatrix = metrics.confusionMatrix
          
          
            // Overall statistics
          System.out.println("Accuracy = " + metrics.accuracy);
          println(s"Weighted precision = ${metrics.weightedPrecision}");
          println(s"Weighted recall = ${metrics.weightedRecall}}" );
          println(s"Weighted F1 score = ${metrics.weightedFMeasure}" );
          println(s"Weighted false positive rate = ${metrics.weightedFalsePositiveRate}" );
          System.out.println("Confusion matrix: \n" + confusionMatrix);
          
          val metrics_len = metrics.labels.length
              // Stats by labels
              //Range(0,metrics.labels.length).
              metrics.labels.toSeq.foreach {i=>
              println(s"Precisiaon ,create and Score for Class ${metrics.labels(i.toInt)}")
              println(s"		Precision = ${metrics.precision(metrics.labels(i.toInt))}");
              println(s"		Recall = ${metrics.recall(metrics.labels(i.toInt))}" );
              println(s"		F1 score = ${metrics.fMeasure(metrics.labels(i.toInt))}\n");
              }
          val numClasses = ovs_prediction.select("prediction").distinct().count()
          // compute the false positive rate per label
         // val predictionColSchema = ovs_prediction.schema("prediction")
          //val numClasses = new MetadataUtils//().getNumClasses(predictionColSchema).get
          val fprs = Range(0, numClasses.toInt).map(p => (p, metrics.falsePositiveRate(p.toDouble)))

  }
  
  def CallOneVsALLAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame, dataSetType: String )={
                      
             import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
              val classifier = new LogisticRegression()
                                      .setRegParam(0.01)
                                      //.setMaxIter(10)
                                      //.setTol(1E-6)
                                      //.setFitIntercept(true)
          
              val ovr = new OneVsRest().setClassifier(classifier)
                 
              val ovs_prediction = ovr.fit(pipedDF).transform(pipedtestDF)
                 ovs_prediction.show()
         
             println("--------------------------------------------------------------Applying General One-vs-Rest classifier (a.k.a. One-vs-All) ------------------------------------------")
             // instantiate the base classifier
            
              ovs_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)
              
              if ( dataSetType == "Binary")
                                   {
                                     //Lets mearsure the accruarcy
                                     Confusion_matrixCalculation(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
                                     BinaryClassificationEvaluator_ROC(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
                                     BinaryClassificationEvaluator_PR(ovs_prediction,"General One-vs-Rest classifier (a.k.a. One-vs-All)")
                                     // instantiate the base classifier
                                  }
              else if (dataSetType == "MultiClass")
              {
                
                MultiClassConfusionMatrix(ovs_prediction)
              }
          } 
  
   def CallGradiantBoosterTreeLAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame )={
  
               import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}      
               val gbt = new GBTClassifier().
                                setMaxIter(10)
               
               val gbt_prediction = gbt.fit(pipedDF).transform(pipedtestDF)
               gbt_prediction.show()
               println("--------------------------------------------------------------Applying General Gradient Booster tree Classfier ------------------------------------------")
               gbt_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)            
               //Lets mearsure the accruarcy
               Confusion_matrixCalculation(gbt_prediction,"General Gradient Booster tree Classfier")
               BinaryClassificationEvaluator_ROC(gbt_prediction,"General Gradient Booster tree Classfier")
               BinaryClassificationEvaluator_PR(gbt_prediction,"General Gradient Booster tree Classfier")
                            
          }
 
  def CallDecisionTreeClassifierLAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame, dataSetType: String  )={
                  
                       //create Decion  tree model
     import org.apache.spark.ml.classification.DecisionTreeClassifier
     val dt_prediction = new DecisionTreeClassifier().fit(pipedDF).transform(pipedtestDF)
     dt_prediction.show()
     println("--------------------------------------------------------------Applying General Decision tree Classfier ------------------------------------------")
      dt_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)      
       if (dataSetType == "MultiClass")
                {
                  
                  MultiClassConfusionMatrix(dt_prediction)
                }
        else {       
             //Lets mearsure the accruarcy
             Confusion_matrixCalculation(dt_prediction,"General Decision tree Classfier")
             BinaryClassificationEvaluator_ROC(dt_prediction,"General Decision tree Classfier")
             BinaryClassificationEvaluator_PR(dt_prediction,"General Decision tree Classfier")
   
            }                 
 
         }
  
    def CallRandomForestClassifierLAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame , dataSetType: String )={
  
                   //create random forest model
            import org.apache.spark.ml.classification.{RandomForestClassifier}
            val rf_prediction = new RandomForestClassifier().fit(pipedDF).transform(pipedtestDF)
            rf_prediction.show()
            println("--------------------------------------------------------------Applying General Random Forest Classfier ------------------------------------------")
            rf_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)  
            if (dataSetType == "MultiClass")
                        {
                          MultiClassConfusionMatrix(rf_prediction)
                        }
                else {       
                     //Lets mearsure the accruarcy
                     Confusion_matrixCalculation(rf_prediction,"General Random Forest Classfier")
                     BinaryClassificationEvaluator_ROC(rf_prediction,"General Random Forest Classfier")
                     BinaryClassificationEvaluator_PR(rf_prediction,"General Random Forest Classfier")
                      }
                     
            }
    
  def CallLogisticRegressionAlgo(pipedDF: org.apache.spark.sql.DataFrame , pipedtestDF : org.apache.spark.sql.DataFrame, dataSetType: String )={
  
       //create random forest model
        import org.apache.spark.ml.classification.{LogisticRegression,LogisticRegressionModel}
        val lr_prediction = new LogisticRegression().fit(pipedDF).transform(pipedtestDF)
        lr_prediction.show()
        println("--------------------------------------------------------------Appying General Logistic Regression ------------------------------------------")
           //create random forest model
        lr_prediction.groupBy(col("prediction"), col("label")).count().orderBy(col("count").desc_nulls_first).show(500)           
        if (dataSetType == "MultiClass")
                {
                  
                  MultiClassConfusionMatrix(lr_prediction)
                }
        else {       
             //Lets mearsure the accruarcy
             Confusion_matrixCalculation(lr_prediction,"General Logistic regression")
             BinaryClassificationEvaluator_ROC(lr_prediction,"General Logistic regression")
             BinaryClassificationEvaluator_PR(lr_prediction,"General Logistic regression")
 
          }                  
      }
  
def getStringIndexersArray(CatalogicalFeatureList:Seq[String]) ={
        CatalogicalFeatureList.map { colName =>new StringIndexer().setInputCol(colName).setOutputCol(colName + "Indexed").setHandleInvalid("keep")
          }.toArray
}

def  CategoricalFeatureVectorslicer( CatalogicalFeatureList:Seq[String]) = {  
             
                   //Array of String indexes of string features
                    import org.apache.spark.ml.feature.VectorSlicer
                    import org.apache.spark.ml.feature.StandardScaler
                    val stringIndexers = getStringIndexersArray(CatalogicalFeatureList)
                   //val OHEncoder = CatalogicalFeatureList.map { colName =>new OneHotEncoderEstimator().setInputCols(Array(colName + "Indexed")).
                     //                                    setOutputCols(Array(colName + "Vector"))}.toArray
                   val slicerArr =  CatalogicalFeatureList.map{colName =>new VectorSlicer().setInputCol(colName + "Indexed").setOutputCol(colName + "Vector")//.setNames(colName + "Indexed")
                                                            }  
                   val scalerArr =  CatalogicalFeatureList.map{colName =>new StandardScaler().setInputCol(colName + "Vector").setOutputCol(colName + "features").setWithStd(true).setWithMean(true)
                                                            }
                    (stringIndexers ++ slicerArr ++ scalerArr)
                   //partial_Stage
                       }


def CategoricalFeatureVectorzing( CatalogicalFeatureList:Seq[String]) = {  
             
                   //Array of String indexes of string features
                   val stringIndexers = getStringIndexersArray(CatalogicalFeatureList)
                   val OHEncoder = CatalogicalFeatureList.map { colName =>new OneHotEncoderEstimator().setInputCols(Array(colName + "Indexed")).setHandleInvalid("keep").
                                                         setOutputCols(Array(colName + "Vector"))}.toArray
                   //val partial_Stage= 
                    (stringIndexers ++ OHEncoder)
                   //partial_Stage
             }
       
def FeatureAssembler( CatalogicalFeatureList:Seq[String] , numbericalFeatureList:Seq[String] ) = {
                   val indexedfeaturesCatColNames = if (!CatalogicalFeatureList.isEmpty) CatalogicalFeatureList.map(_ + "Vector") else Seq()
                   val allIndexedFeaturesColNames = if  (CatalogicalFeatureList.isEmpty && !numbericalFeatureList.isEmpty) 
                                                                numbericalFeatureList
                                                   else if (!CatalogicalFeatureList.isEmpty && numbericalFeatureList.isEmpty )
                                                                 indexedfeaturesCatColNames
                                                   else  numbericalFeatureList ++ indexedfeaturesCatColNames
                     
             
                   //feature Assembler that contains the dense sparse matrix of featured columnns
                    val assembler = new VectorAssembler().setInputCols(Array(allIndexedFeaturesColNames: _*)).setOutputCol("features").setHandleInvalid("skip")
                   Array(assembler)
             }
     
  def CallOneVsALLFullAlgo(traningDF: org.apache.spark.sql.DataFrame , testingDF : org.apache.spark.sql.DataFrame, dataSetType: String,featureStringCol: Seq[String] , featureNumericalCol: Seq[String] )={
  
             import org.apache.spark.ml.{Pipeline,Transformer,PipelineModel}
             
             val stages = CategoricalFeatureVectorzing(featureStringCol) ++ FeatureAssembler(featureStringCol,featureNumericalCol) 
               
              // pipelinedStages              
             val pipelinedStages = new Pipeline().setStages(stages)
             
              
             // create piped train DF
             val pipedDF = pipelinedStages.fit(traningDF).transform(traningDF)
             
             println("pipedDF training DF")
             pipedDF.show(false)
             
             // create piped test DF
             val pipedtestDF = pipelinedStages.fit(testingDF).transform(testingDF)
             println("pipedtestDF testing DF")
             pipedtestDF.show(false)  
            
             ML_scalaAdvanceMethods.CallOneVsALLAlgo(pipedDF,pipedtestDF,dataSetType)
              
          }
            
 def textCleaningDf (rawDF: org.apache.spark.sql.DataFrame, ContentColname : String  )={
   
   val punctuation = "[^a-zA-Z0-9]"
   val digits = "\\b\\d+\\b"
   val white_space = "\\s+"
   val small_words = "\\b[a-zA-Z0-9]{1,2}\\b"
   val urls = "(https?\\://)\\S+"
   val cleanedDF = rawDF.withColumn(ContentColname+"1", regexp_replace(col(ContentColname), punctuation," ") ).
         withColumn(ContentColname+"2", regexp_replace(col(ContentColname+"1"), digits,"") ).
         withColumn(ContentColname+"3", regexp_replace(col(ContentColname+"2"), small_words,"") ).
         withColumn(ContentColname+"4", regexp_replace(col(ContentColname+"3"), urls,"") ).
         withColumn(ContentColname+"Cleaned", regexp_replace(col(ContentColname+"4"), white_space," ") ).
         drop(ContentColname+"1").
         drop(ContentColname+"2").
         drop(ContentColname+"3").
         drop(ContentColname+"4")
  
  
   cleanedDF.select(ContentColname+"Cleaned").show(2,false)
  // Removing Stop words Loading a stopwords list
   //var Stopwords = Map[String, List[String]]()
   //Stopwords += ("english" -> Source.fromFile("stopwords.txt").getLines().toList)
//    import org.apache.spark.ml.feature.StopWordsRemover
//    val remover = new StopWordsRemover()
//    .setInputCol(ContentColname+"Cleaned")
//    .setOutputCol("filtered")
//   val cleanedDF2 = remover.transform(cleanedDF)
   cleanedDF
  //Function to perform step by step text preprocessing and cleaning on documents 
   //def cleanDocument(document_text: String) : String = {
   
       // Building a List of Regex for PreProcessing the text 
       //    var RegexList = Map[String, String]()
       //    RegexList += ("punctuation" -> punctuation)
       //    RegexList += ("digits" -> digits)
       //    RegexList += ("white_space" -> white_space)
       //    RegexList += ("small_words" -> small_words)
       //    RegexList += ("urls" -> urls)
       //Converting all words to lowercase
       // Removing URLs from document
       // Removing Punctuations from document text
       // Removing Digits from document text
       // Removing all words with length less than or equal to 2
       // Removing extra whitespaces from text
       // Removing English Stopwords
       // Returning the preprocessing and cleaned document text
    
       // var text = document_text.toLowerCase
       // text = removeRegex(text,"urls")
       // text = removeRegex(text,"punctuation")
       // text = removeRegex(text,"digits")
       // text = removeRegex(text,"small_words")
       // text = removeRegex(text,"white_space")
       // text = removeCustomWords(text, "english")
   //}  
    
 }
 
   
        
}