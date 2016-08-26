name := "LambdaMART"

version := "3.0"

scalaVersion := "2.10.6"
scalaBinaryVersion := "2.10"

dependencyOverrides ++= Set(
  "org.scala-lang" % "scala-library" % scalaVersion.value,
  "org.scala-lang" % "scala-compiler" % scalaVersion.value,
  "org.scala-lang" % "scala-reflect" % scalaVersion.value
)

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.6.0" % "provided",
  "org.apache.spark" % "spark-mllib_2.10" % "1.6.0" % "provided",
  "com.github.scopt" % "scopt_2.10" % "3.3.0"
)
