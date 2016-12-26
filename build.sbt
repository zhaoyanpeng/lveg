import Version._

lazy val commonSettings = Seq(
  organization := "edu.shanghaitech.ai.nlp",
  version := "0.0.1",
  
  scalaVersion := Version.scala,
  crossScalaVersions := Seq("2.12.0", "2.10.4"),
  
  libraryDependencies ++= Seq(
	Library.log4j, 
	Library.junit,
	Library.BerkeleyParser
  ),
  javacOptions ++= Seq("-source", "1.8"),
  javaOptions += "-Xlint:unchecked",
  javaOptions += "-Xmx10g",
  fork := true
)

lazy val root = project
  .in(file("."))
  .settings(commonSettings: _*)
  .settings(
	name := "lveg"
  )