package com.packt.scalada.viz.breeze

import io.continuum.bokeh._
import math.{Pi=>pi,sin}


object bokesh_circle extends App with Tools{
  val pi = 3.14
  object source extends ColumnDataSource {
        val x = column(-2*pi to 2*pi by 0.1)
        val y = column(x.value.map(sin))
    }

    import source.{x,y}
    //1) create plot 
    val plot = new Plot().title("cirlce glyph")
   
    //2) Let's create a marker object to mark the data point 
    val glyph = new Circle().x(x).y(y).size(5).fill_color(Color.Red).line_color(Color.Black)
    
    //2.1) Let's compose the main graph
    val circle = new GlyphRenderer().data_source(source).glyph(glyph)

    //3) Set Data range for the X and the Y Axis
    val xdr = new DataRange1d()
    val ydr = new DataRange1d()
    
   
    //3.1 fit the ranges to the plot
    plot.x_range(xdr).y_range(ydr).tools(Pan|WheelZoom|Reset|Resize|PreviewSave|Crosshair)

    //4) X and Y linear Axis
    val xaxis = new LinearAxis().plot(plot).location(Location.Below)
    val yaxis = new LinearAxis().plot(plot).location(Location.Left)
    plot.below <<= (xaxis :: _)
    plot.left <<= (yaxis :: _)

   //4.1) Render X and Y Axis of plot
    plot.renderers := List(xaxis, yaxis, circle)
   // skip grid creation
    // skip legend creation and render legend
    //8) document creation from plot
    val document = new Document(plot)
    // 9) save plot in html format 
    val html = document.save("sample.html")
    println(s"Wrote ${html.file}. Open ${html.url} in a web browser.")
    html.view()
}