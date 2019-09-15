/*
 * iris.data is read using scala.io.Source
 * Doing string column to numeric column transformation
 * Strong data as irisNumericCsvFile using java.io.FileWriter
 */

package com.packt.scalada.viz.breeze

import scala.io.Source
import breeze.linalg._
import scala.reflect.io.File
import java.io.PrintWriter
import java.io.FileWriter

object PrepareIrisDataApp extends App{
  
  def fnTransformSpecies(record:Array[String]):Array[String]={
    val modified=if (record(4).equalsIgnoreCase("Iris-setosa")) record.updated(4, "0")
    			else if (record(4).equalsIgnoreCase("Iris-versicolor")) record.updated(4, "1")
    			else if (record(4).equalsIgnoreCase("Iris-virginica")) record.updated(4, "2")
    			else Array.empty[String]
   
    modified
  }
   
  println("read  iris.data File ")
  val irisPreprocessed=for {
    eachLine<-Source.fromFile("iris.data").getLines()
    tfmdLineIter=fnTransformSpecies(eachLine.split(","))
  }yield (tfmdLineIter)
 
  
  /**You could actually proceed to create a Matrix out of that array.**/
  
  /*val denseMatrix=DenseMatrix(irisPreprocessed.toArray: _*)
  println (denseMatrix(0 to 5, ::))*/
  
  
  val irisNumericCsvFile=new PrintWriter(new FileWriter("irisNumeric.csv"))
  irisPreprocessed.foreach{ lineArray=>
    irisNumericCsvFile.println(lineArray.mkString(","))
  }
  println("create irisNumericCsv File ")
  
  irisNumericCsvFile.flush()
  irisNumericCsvFile.close()

}