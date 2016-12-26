import sbt._

object Version {
  val scala                = "2.12.0"
  val junit                = "4.12"
  val log4j                = "1.2.16"
  val BerkeleyParser       = "1.7"
}

object Library {
  var junit                = "junit"                % "junit"                % Version.junit
  var log4j                = "log4j"                % "log4j"                % Version.log4j
  val BerkeleyParser       = "BerkeleyParser"       % "BerkeleyParser"       % Version.BerkeleyParser        from "https://raw.githubusercontent.com/slavpetrov/berkeleyparser/master/BerkeleyParser-1.7.jar"
}