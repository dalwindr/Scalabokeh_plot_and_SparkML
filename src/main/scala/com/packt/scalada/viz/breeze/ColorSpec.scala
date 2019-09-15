package com.packt.scalada.viz.breeze
import io.continuum.bokeh
import io.continuum.bokeh._
import breeze.linalg.{DenseMatrix,DenseVector,linspace}
import org.apache.log4j._
import java.lang.System
import org.joda.time.{LocalDate=>Date}
//https://github.com/bokeh/bokeh-scala/blob/master/examples/src/main/scala/models/ColorSpec.scala
object ColorSpec extends App {
  
  object source extends ColumnDataSource {
        val xAxis     = column(Seq(1, 2, 3, 4, 5)) // DenseVector(1, 2, 3, 4, 5)
        val yAxis     = column(Seq(5, 4, 3, 2, 1))
        val color = column(Seq(RGB(0, 100, 120), Color.Green, "#2c7fb8": Color, RGBA(120, 230, 150, 0.5)))
    }

    import source.{xAxis,yAxis,color}

    val xdr = new DataRange1d()
    val ydr = new DataRange1d()

    val circle = new Circle().x(44).y(55).size(15).fill_color(color).line_color(Color.Black)

    val renderer = new GlyphRenderer()
        .data_source(source)
        .glyph(circle)

    val plot = new Plot().x_range(xdr).y_range(ydr)

    val xaxis = new DatetimeAxis().plot(plot)
    val yaxis = new LinearAxis().plot(plot)
    plot.below <<= (xaxis :: _)
    plot.left <<= (yaxis :: _)

    val pantool = new PanTool().plot(plot)
    val wheelzoomtool = new WheelZoomTool().plot(plot)

    plot.renderers := List(xaxis, yaxis, renderer)
    plot.tools := List(pantool, wheelzoomtool)

    val document = new Document(plot)
    val html = document.save("colorspec.html")
    println(s"Wrote ${html.file}. Open ${html.url} in a web browser.")
    html.view
}