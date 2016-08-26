package org.apache.spark.mllib.tree.model

import org.apache.spark.mllib.tree.configuration.FeatureType

class SplitInfo(feature: Int, threshold: Double)
  extends Split(feature, threshold, FeatureType.Continuous, List())
