package com.packt.scalada.viz.breeze

import io.continuum.bokeh
import io.continuum.bokeh._
import breeze.linalg.{DenseMatrix,DenseVector,linspace}
import org.apache.log4j._
import java.lang.System


//import thirdparty._

object anacombeBokeh extends App{
    val anscombe_quartet = DenseMatrix(
        (10.0,  8.04, 10.0, 9.14, 10.0,  7.46,  8.0,  6.58),
        ( 8.0,  6.95,  8.0, 8.14,  8.0,  6.77,  8.0,  5.76),
        (13.0,  7.58, 13.0, 8.74, 13.0, 12.74,  8.0,  7.71),
        ( 9.0,  8.81,  9.0, 8.77,  9.0,  7.11,  8.0,  8.84),
        (11.0,  8.33, 11.0, 9.26, 11.0,  7.81,  8.0,  8.47),
        (14.0,  9.96, 14.0, 8.10, 14.0,  8.84,  8.0,  7.04),
        ( 6.0,  7.24,  6.0, 6.13,  6.0,  6.08,  8.0,  5.25),
        ( 4.0,  4.26,  4.0, 3.10,  4.0,  5.39, 19.0, 12.50),
        (12.0, 10.84, 12.0, 9.13, 12.0,  8.15,  8.0,  5.56),
        ( 7.0,  4.82,  7.0, 7.26,  7.0,  6.42,  8.0,  7.91),
        ( 5.0,  5.68,  5.0, 4.74,  5.0,  5.73,  8.0,  6.89))

    object circles1 extends ColumnDataSource {
      
        val xi  =  column(anscombe_quartet(::, 0))
        val yi   = column(anscombe_quartet(::, 1))
        
        val xii  = column(anscombe_quartet(::, 2))
        val yii  = column(anscombe_quartet(::, 3))
        
        val xiii = column(anscombe_quartet(::, 4))
        val yiii = column(anscombe_quartet(::, 5))
        
        val xiv  = column(anscombe_quartet(::, 6))
        val yiv  = column(anscombe_quartet(::, 7))
        
       
    }
    object lines extends ColumnDataSource {
        val x = column(linspace(-0.5, 20.5, 10))
        val y = column(x.value*0.5 + 3.0)
    }


    def make_plot(title: String, xPos: Int, yPos: Int ) = {
        
        //1. create plot
        val plot = new Plot().title("dalwinder test " + title)//.tools(Pan|WheelZoom|Reset|Resize|PreviewSave|Crosshair)
        
        val circle_axis = Array(circles1.xi,circles1.yi,circles1.xii,circles1.yii,circles1.xiii,circles1.yiii,circles1.xi,circles1.xiv,circles1.yiv)
        
        val x_col= circle_axis(xPos)
        val y_col = circle_axis(yPos)
        //2 choose circle as glyph for plotting
        val  glyphCircle= new Circle().x(x_col).y(y_col).size(12).fill_color("#cc6633").line_color("#cc6633").fill_alpha(50%%)
        //val glyph = new Circle().x(x).y(y).size(5).fill_color(Color.Red).line_color(Color.Black)
        
        //2.1 render circle glyph using  GlyphRenderer()
        val circle_renderer = new GlyphRenderer().data_source(circles1).glyph(glyphCircle)
        
        
        //3 Define ranges
        val xdr = new Range1d().start(-0.5).end(20.5)
        val ydr = new Range1d().start(-0.5).end(20.5)

        //3.1 fit ranges to the plot
        plot.x_range(xdr).y_range(ydr)
                            .width(400).height(400).border_fill(Color.White)
                            .background_fill("#e9e0db") //.tools(Pan|WheelZoom|Reset|Resize|PreviewSave|Crosshair)

        // 4 define x and y linear Axis of plot
        val xaxis = new LinearAxis().plot(plot).axis_line_color()
        val yaxis = new LinearAxis().plot(plot).axis_line_color()
        // 4.1  set the location of x  and y Axis
        plot.below <<= (xaxis :: _)
        plot.left <<= (yaxis :: _)
       
        // 5. defin the grid for X and y axis of the plot
        val xgrid = new Grid().plot(plot).axis(xaxis).dimension(0)
        val ygrid = new Grid().plot(plot).axis(yaxis).dimension(1)
        
        // 2  find the the lineGlyph
        val lineGlyph = new Line().x(lines.x).y(lines.y).line_color("#666699").line_width(2)
        
        // 2.1 rnd the lineGlyph using GlyphRenderer
        val line_renderer = new GlyphRenderer().data_source(lines).glyph(lineGlyph)
        
        //6 Tools
        val panTool = new PanTool().plot(plot)
        val wheelZoomTool = new WheelZoomTool().plot(plot)
        val previewSaveTool = new PreviewSaveTool().plot(plot)
        val resetTool = new ResetTool().plot(plot)
        val resizeTool = new ResizeTool().plot(plot)
        val crosshairTool = new CrosshairTool().plot(plot)
        
        plot.tools := List(panTool, wheelZoomTool, previewSaveTool, resetTool, resizeTool, crosshairTool)
        
        // 6. find render everything    
        plot.renderers := List(xaxis, yaxis, xgrid, ygrid, line_renderer, circle_renderer)
        plot
    }
    
   // type mismatch; found : io.continuum.bokeh.ColumnDataSource#Column[breeze.linalg.DenseVector,Double] 
    //required: com.packt.scalada.viz.breeze.anacombeBokeh.circles1.Column[breeze.linalg.DenseVector,Double]
    //io.continuum.bokeh.ColumnDataSource.Column

        

    val I   = make_plot("I",  0,   1)
    val II  = make_plot("II",  2,   3)
    val III =  make_plot("III",  4,   5)
    val IV  =  make_plot("IV",  6,   7)

    val children = List(List(I, II), List(III, IV))
    val grid = new GridPlot().children(children).width(800)

    val document = new Document(grid)
    val html = document.save("anscombe.html")
    
    println(s"Wrote ${html.file}. Open ${html.url} in a web browser.")
    html.view()
}
