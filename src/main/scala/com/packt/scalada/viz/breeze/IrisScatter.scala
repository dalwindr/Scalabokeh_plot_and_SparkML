/*
 * It picks of 3,4,5 columns of the file and creating plots using bokeh, and breeze
 * 
 * 
 */

package com.packt.scalada.viz.breeze

import io.continuum.bokeh.ColumnDataSource
import breeze.linalg._  //It will allow to read csv file using csvread into matrix like matlab 
import java.io.File
import io.continuum.bokeh.Color
import io.continuum.bokeh.Plot
import io.continuum.bokeh.Document
import io.continuum.bokeh.GlyphRenderer
import io.continuum.bokeh.Diamond
import io.continuum.bokeh.DataRange1d
import io.continuum.bokeh.LinearAxis
import io.continuum.bokeh._
import io.continuum.bokeh.DataRange1d

//vegas vs jfx vs bokeh vs plotly scala vs swift vis2

object BreezeSource {

  val iris = csvread(file = new File("irisNumeric.csv"), separator = ',')
  println(iris)
  val colormap = Map[Int, Color](0 -> Color.Red, 1 -> Color.Green, 2 -> Color.Blue)
  println(colormap)
  object Diamondsource11 extends ColumnDataSource {
    val sepal_length = column(iris(::, 0)) //First column, all rows
    val  sepal_width = column(iris(::, 1))
    val petal_length = column(iris(::, 2))
    val petal_width = column(iris(::, 3))
    val species = column(iris(::, 4))
    println("\n sepal_length = " + sepal_length)
    println("\n sepal_width = " + sepal_width)
    
    println("\n petal_length = " + petal_length)
    println("\n petal_width = " + petal_width)
    println("\n species = " + species)
    
    val speciesColor = column(iris(::, 4).map(each=>colormap(each.toInt)))
    println("\n speciesColor = " + speciesColor)
  }

}


object IrisScatter extends App {
  //First lets load our iris data 
  import BreezeSource._
  import Diamondsource11._

   //1) Create the plot with the title "Petal length vs Width" 
  val plot = new Plot().title("Iris Petal Length vs Petal Width")

  //2) Let's create a marker object to mark the data point 
  val diamond = new Diamond()
    .x(petal_length)
    .y(petal_width)
    .fill_color(speciesColor)
    .fill_alpha(0.5)
    .size(10)

  //2.1) Let's compose the main graph
  val dataPointRenderer = new GlyphRenderer().data_source(Diamondsource11).glyph(diamond)

  //3) Set Data range for the X and the Y  Axis
  val xRange = new DataRange1d().sources(petal_length :: Nil)
  val yRange = new DataRange1d().sources(petal_width :: Nil)
  //3.1 fit the ranges to the plot
  plot.x_range(xRange).y_range(yRange)

  //4) X and Y linear Axis
  val xAxis = new LinearAxis().plot(plot).axis_label("Petal Length").bounds((1.0, 7.0))
  val yAxis = new LinearAxis().plot(plot).axis_label("Petal Width").bounds((0.0, 2.5))
 
  
  //4.1) Render X and Y Axis of plot
  plot.below <<= (listRenderer => (xAxis :: listRenderer))
  plot.left <<= (listRenderer => (yAxis :: listRenderer))
  
  //5) create Grid over X and Y axis
  val xgrid = new Grid().plot(plot).axis(xAxis).dimension(0)
  val ygrid = new Grid().plot(plot).axis(yAxis).dimension(1)
  
  //6) Tools
  val panTool = new PanTool().plot(plot)
  val wheelZoomTool = new WheelZoomTool().plot(plot)
  val previewSaveTool = new PreviewSaveTool().plot(plot)
  val resetTool = new ResetTool().plot(plot)
  val resizeTool = new ResizeTool().plot(plot)
  val crosshairTool = new CrosshairTool().plot(plot)
  
  //7.1) Legends - Manual :-(
  val setosa = new Diamond().fill_color(Color.Red).size(10).fill_alpha(0.5)
  val setosaGlyphRnd=new GlyphRenderer().glyph(setosa)
  val versicolor = new Diamond().fill_color(Color.Green).size(10).fill_alpha(0.5)
  val versicolorGlyphRnd=new GlyphRenderer().glyph(versicolor)
  val virginica = new Diamond().fill_color(Color.Blue).size(10).fill_alpha(0.5)
  val virginicaGlyphRnd=new GlyphRenderer().glyph(virginica)
  
  val legends = List("setosa" -> List(setosaGlyphRnd),
		  			"versicolor" -> List(versicolorGlyphRnd),
		  			"virginica" -> List(virginicaGlyphRnd))
		  			
  val legend = new Legend().orientation(LegendOrientation.TopLeft).plot(plot).legends(legends)
  
 //8.2 ) Add the renderers and the tools to the plot
  plot.renderers := List(xAxis, yAxis, dataPointRenderer, xgrid, ygrid, legend, setosaGlyphRnd, virginicaGlyphRnd, versicolorGlyphRnd)
  plot.tools := List(panTool, wheelZoomTool, previewSaveTool, resetTool, resizeTool, crosshairTool)
  
  //8) document creation from plot
  val document = new Document(plot)
  
  // 9) save plot in html format 
  val file = document.save("IrisBokehBreeze.html")

  println(s"Saved the chart as ${file.url}")
}

