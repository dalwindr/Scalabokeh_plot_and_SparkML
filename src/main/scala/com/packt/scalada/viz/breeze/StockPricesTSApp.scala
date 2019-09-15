/*
 * 1) reading stock csv from dow_jones_index.data
 * 2) create temp table
 * 3) create four plot on four different values of column 3 ( "MSFT","BAC","CAT","MMM")
 */

package com.packt.scalada.viz.breeze

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import com.databricks.spark.csv._
import io.continuum.bokeh._
import org.joda.time.format.DateTimeFormat
import org.apache.log4j._
import java.lang.System
import org.apache.spark.sql.SparkSession 
import org.json4s.Serialization

object StockPricesTSApp {

  def main(args: Array[String]) {
     Logger.getLogger("org").setLevel(Level.ERROR)
	  val spark = SparkSession.builder().config("spark.master","local").appName("Chapter 3 Scala cookbook").getOrCreate()
	  val stocks = spark.read.format("csv").option("header", "true").option("Delimiter",",").load("dow_jones_index.data")
	  stocks.groupBy("stock").count().show()
	  stocks.createOrReplaceTempView("stocks")
	
	  // It returns array of Array(dateData) and Array(closeData) w.r.t stock  "MSFT"  with color Blue 
	  val microsoftPlot = plotWeeklyClosing(spark.sqlContext, "MSFT", Color.Blue) 
	  
	  // It returns array of Array(dateData) and Array(closeData) w.r.t stock  "BAC"  with color  Red 
	  val bofaPlot = plotWeeklyClosing(spark.sqlContext, "BAC", Color.Red)
	  
	  // It returns array of Array(dateData) and Array(closeData) w.r.t stock  "BAC"  with color  Red 
	  val caterPillarPlot = plotWeeklyClosing(spark.sqlContext, "CAT", Color.Orange)
	  
	  // It returns array of Array(dateData) and Array(closeData) w.r.t stock  "BAC"  with color  Red 
	  val mmmPlot = plotWeeklyClosing(spark.sqlContext, "MMM", Color.Black)
	
	  val msDocument = new Document(microsoftPlot)
	  val msHtml = msDocument.save("MicrosoftClosingPrices.html")
	
	  println(s"Saved the Microsoft Chart as ${msHtml.url}")
	  
	  
	  val children = List(List(microsoftPlot, bofaPlot), List(caterPillarPlot, mmmPlot))
	  val grid = new GridPlot().children(children)
	  
	  // 
	  val document = new Document(grid)
    
    //Storing grid document in html
	  val html = document.save("DJClosingPrices.html")
	  
	  println(s"Saved 4 Grid stock chart as ${html.url}")
	  html.view
	  
  }
  
  def plotWeeklyClosing(sqlContext:SQLContext, ticker: String, color: Color) = {

    val source = StockSource.getSource(ticker, sqlContext)
    
    import source._

    //1) Create Plot
    val plot = new Plot().title(ticker.toUpperCase()).width(800).height(400)
    

    //2) using Line plot instead of diamond
    // Let's create a marker object to mark the data point 
    val line = new Line().x(date).y(close).line_color(color).line_width(2)
    
    //2.1) lets compose main graph using GlyphRenderer 
    val lineGlyph = new GlyphRenderer().data_source(source).glyph(line)
    
   //3) Fixing Data ranges for column Date and Close
   //Set Data range for the X and the Y Axis
    val xdr = new DataRange1d().sources(List(date))
    val ydr = new DataRange1d().sources(List(close))
    
    //3.1 fit the ranges to the plot
    plot.x_range(xdr).y_range(ydr)

    
    // create X and Y axis using for column Month and Price w .r.t date and close
    val xformatter = new DatetimeTickFormatter().formats(Map(DatetimeUnits.Months -> List("%b %Y")))
    // X axis on DataTime Axis with datatime formater
    val xaxis = new DatetimeAxis().plot(plot).formatter(xformatter).axis_label("Month")
     
    // Y axis on Linear Axis
    val yaxis = new LinearAxis().plot(plot).axis_label("Price")
    
    //4.1) Render X and Y Axis of plot
    plot.below <<= (xaxis :: _)
    plot.left <<= (yaxis :: _)
    
    // 5 . creating grid over x axis and y axis
    val xgrid = new Grid().plot(plot).dimension(0).axis(xaxis)
    val ygrid = new Grid().plot(plot).dimension(1).axis(yaxis)

    //6. creating Tools
    val panTool = new PanTool().plot(plot)
    val wheelZoomTool = new WheelZoomTool().plot(plot)
    val previewSaveTool = new PreviewSaveTool().plot(plot)
    val resetTool = new ResetTool().plot(plot)
    val resizeTool = new ResizeTool().plot(plot)
    val crosshairTool = new CrosshairTool().plot(plot)

    //7. creating Legend
    val legends = List(ticker -> List(lineGlyph))
    val legend = new Legend().plot(plot).legends(legends)

    //8. Adding renderers and Tools
    plot.renderers <<= (xaxis :: yaxis :: xgrid :: ygrid :: lineGlyph :: legend :: _)
    plot.tools := List(panTool, wheelZoomTool, previewSaveTool, resetTool, resizeTool, crosshairTool)

    plot

  }

}

object StockSource {
  
  val formatter = DateTimeFormat.forPattern("MM/dd/yyyy");
  
  def getSource(ticker: String, sqlContext: SQLContext) = {
    val stockDf = sqlContext.sql(s"select stock, date, close from stocks where stock= '$ticker'")
    stockDf.cache()
    stockDf.show()
    
    val dateData: Array[Double] = stockDf.select("date").collect.map(eachRow => formatter.parseDateTime(eachRow.getString(0)).getMillis().toDouble)
    val closeData: Array[Double] = stockDf.select("close").collect.map(eachRow => eachRow.getString(0).drop(1).toDouble)

    println("\n dateData :" + dateData.toList )
    println("\n closeData :" + closeData.toList )
    
    object source extends ColumnDataSource {
      val date = column(dateData)
      val close = column(closeData)
    }
    source
  }
}