package org.apache.spark.mllib.tree.model

import org.apache.spark.broadcast.Broadcast

object nodePredict {

	def predict(nodeId: Int, 
				nodeIdTracker: Broadcast[Array[Byte]], 
				lambdas: Broadcast[Array[Double]], 
				weights: Broadcast[Array[Double]]): Double = {
		var lambdaSum = 0.0
		var weightSum = 0.0

		var sampleIdx = 0
		while(sampleIdx < nodeIdTracker.value.length) {
			if(nodeId == nodeIdTracker.value(sampleIdx)){
				lambdaSum += lambdas.value(sampleIdx)
				weightSum += weights.value(sampleIdx)
			}
			sampleIdx += 1
		}

		var leafValue = lambdaSum/weightSum
		leafValue
	}
		//def adjustPredict()
}