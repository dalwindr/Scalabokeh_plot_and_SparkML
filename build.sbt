organization := "com.packt"

name := "chapter4-visualization"

scalaVersion := "2.11.12"
val breezeVersion = "0.12"
val sparkVersion="2.4.0"
val bokehVersion="0.5"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % breezeVersion,
  "org.scalanlp" %% "breeze-natives" % breezeVersion,
  "org.scalanlp" %% "breeze-macros" % breezeVersion, 
  "org.scalanlp" %% "breeze-viz" % breezeVersion,
  "io.continuum.bokeh" %% "bokeh" % bokehVersion,
  "joda-time" % "joda-time" % "2.6",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.databricks" %% "spark-csv" % "1.0.3"
)
