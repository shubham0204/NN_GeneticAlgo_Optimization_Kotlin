/*
 * Copyright 2020 Shubham Panchal
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package NeuralNetwork

class Model( private var inputDims : Int , var layers : Array<Dense> , var learningRate : Double ) {

    private var y : Matrix? = null
    private var y_hat : Matrix? = null

    fun compile() {
        var inputDimForLayer = inputDims
        for( i in layers.indices ){
            layers[ i ].initWeights( inputDimForLayer )
            inputDimForLayer = layers[ i ].units
        }
    }

    fun forward( inputs : Matrix , labels : Matrix ) : Matrix {
        var layerInput = inputs
        for ( i in layers.indices ){
            val theta = layers[ i ].forward( layerInput )
            layerInput = theta
        }
        y = layerInput
        y_hat = labels
        return y!!
    }

    fun backward() {
        layers.reverse()
        val dJ_dyN = lossGradient( y!! , y_hat!! )
        layers.forEach{ layer -> layer.computeGradients() }
        var dJ_dtheta : Matrix?
        var dtheta_dX : Matrix?
        val dyN_dthetaN = layers[ 0 ].dy_dtheta
        val dthetai_dwi = layers[ 0 ].dtheta_dW
        val dJ_dwN = MatrixOps.dot( dthetai_dwi!!.transpose() , ( dJ_dyN * dyN_dthetaN ) )
        layers[ 0 ].W = optimize( dJ_dwN , layers[ 0 ].W!! )
        layers[ 0 ].B = optimize( dJ_dyN * dyN_dthetaN , layers[ 0 ].B )
        dJ_dtheta = dJ_dyN * dyN_dthetaN
        dtheta_dX = layers[ 0 ].dtheta_dX
        for ( i in 1 until layers.size ) {
            val dJ_dyi = MatrixOps.dot( dJ_dtheta!! , dtheta_dX!!.transpose() )
            val dJ_dthetai = dJ_dyi * layers[ i ].dy_dtheta
            val dJ_dwi = MatrixOps.dot( layers[ i ].dtheta_dW!!.transpose() , dJ_dthetai )
            layers[ i ].W = optimize( dJ_dwi , layers[ i ].W!! )
            layers[ i ].B = optimize( dJ_dthetai , layers[ i ].B )
            dJ_dtheta = dJ_dthetai
            dtheta_dX = layers[ i ].dtheta_dX
        }
        layers.reverse()
    }

    private fun optimize(grad : Matrix, param : Matrix ) : Matrix {
        return param - ( grad * learningRate )
    }

    private fun lossGradient( y : Matrix , y_hat : Matrix ) : Matrix {
        return ( y - y_hat )
    }

}
