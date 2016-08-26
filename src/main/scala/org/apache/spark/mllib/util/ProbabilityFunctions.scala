package org.apache.spark.mllib.util

object ProbabilityFunctions{
  //probit function
  val ProbA = Array( 3.3871328727963666080e0, 1.3314166789178437745e+2, 1.9715909503065514427e+3, 1.3731693765509461125e+4,
    4.5921953931549871457e+4, 6.7265770927008700853e+4, 3.3430575583588128105e+4, 2.5090809287301226727e+3)
  val ProbB = Array(4.2313330701600911252e+1, 6.8718700749205790830e+2, 5.3941960214247511077e+3, 2.1213794301586595867e+4,
    3.9307895800092710610e+4, 2.8729085735721942674e+4, 5.2264952788528545610e+3)

  val ProbC = Array(1.42343711074968357734e0, 4.63033784615654529590e0, 5.76949722146069140550e0, 3.64784832476320460504e0,
    1.27045825245236838258e0, 2.41780725177450611770e-1, 2.27238449892691845833e-2, 7.74545014278341407640e-4)
  val ProbD = Array(2.05319162663775882187e0, 1.67638483018380384940e0, 6.89767334985100004550e-1, 1.48103976427480074590e-1,
    1.51986665636164571966e-2, 5.47593808499534494600e-4, 1.05075007164441684324e-9)

  val ProbE = Array(6.65790464350110377720e0, 5.46378491116411436990e0, 1.78482653991729133580e0, 2.96560571828504891230e-1,
    2.65321895265761230930e-2, 1.24266094738807843860e-3, 2.71155556874348757815e-5, 2.01033439929228813265e-7)
  val ProbF = Array(5.99832206555887937690e-1, 1.36929880922735805310e-1, 1.48753612908506148525e-2, 7.86869131145613259100e-4,
    1.84631831751005468180e-5, 1.42151175831644588870e-7, 2.04426310338993978564e-15)

  def Probit(p: Double): Double ={
    val q = p - 0.5
    var r = 0.0
    if(scala.math.abs(q) < 0.425){
      r = 0.180625 - q * q
      q * (((((((ProbA(7) * r + ProbA(6)) * r + ProbA(5)) * r + ProbA(4)) * r + ProbA(3)) * r + ProbA(2)) * r + ProbA(1)) * r + ProbA(0)) /
      (((((((ProbB(6) * r + ProbB(5)) * r + ProbB(4)) * r + ProbB(3)) * r + ProbB(2)) * r + ProbB(1)) * r + ProbB(0)) * r + 1.0)
    }
    else{
      if(q < 0) r = p
      else  r = 1 - p
      r = scala.math.sqrt( -scala.math.log(r))
      var retval = 0.0
      if(r < 5){
        r = r - 1.6
        retval = (((((((ProbC(7) * r + ProbC(6)) * r + ProbC(5)) * r + ProbC(4)) * r + ProbC(3)) * r + ProbC(2)) * r + ProbC(1)) * r + ProbC(0)) /
        (((((((ProbD(6) * r + ProbD(5)) * r + ProbD(4)) * r + ProbD(3)) * r + ProbD(2)) * r + ProbD(1)) * r + ProbD(0)) * r + 1.0)
      }
      else{
        r = r - 5
        retval = (((((((ProbE(7) * r + ProbE(6)) * r + ProbE(5)) * r + ProbE(4)) * r + ProbE(3)) * r + ProbE(2)) * r + ProbE(1)) * r + ProbE(0)) /
        (((((((ProbF(6) * r + ProbF(5)) * r + ProbF(4)) * r + ProbF(3)) * r + ProbF(2)) * r + ProbF(1)) * r + ProbF(0)) * r + 1.0)
      }
      if(q >= 0) retval else -retval
    }
  }


  //The approximate complimentary error function (i.e., 1-erf).
  def erfc(x: Double): Double = {
    if (x.isInfinity) {
      if(x.isPosInfinity) 1.0 else -1.0
    }
    else {
      val p = 0.3275911
      val a1 = 0.254829592
      val a2 = -0.284496736
      val a3 = 1.421413741
      val a4 = -1.453152027
      val a5 = 1.061405429

      val t = 1.0 / (1.0 + p * math.abs(x))
      val ev = ((((((((a5 * t) + a4) * t) + a3) * t) + a2) * t + a1) * t) * scala.math.exp(-(x * x))
      if (x >= 0) ev else (2-ev)
    }
  }
}