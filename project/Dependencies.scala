import sbt._

object Version {
  val scala                = "2.12.0"
  val junit                = "4.12"
  val log4j                = "1.2.16"
  val pool2                = "2.4.2"
  val BerkeleyParser       = "1.7"
}

object Library {
  val junit                = "junit"                % "junit"                % Version.junit
  val log4j                = "log4j"                % "log4j"                % Version.log4j
  val pool2                = "org.apache.commons"   % "commons-pool2"        % Version.pool2
  val BerkeleyParser       = "BerkeleyParser"       % "BerkeleyParser"       % Version.BerkeleyParser        from "https://raw.githubusercontent.com/slavpetrov/berkeleyparser/master/BerkeleyParser-1.7.jar"
}