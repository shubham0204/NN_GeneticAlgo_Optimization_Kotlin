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

class Dense(
    var units : Int ,
    var activation : Activation,
    var requiresBias : Boolean = false
) {

    // Inputs to this layer
    var X : Matrix? = null
    // Weights
    var W : Matrix? = null
    // Biases
    var B : Matrix = MatrixOps.onesLike( 1 , units )
    // The output this layer will produce
    var y : Matrix? = null
    // The value of WX + B
    var theta : Matrix? = null
    // Some gradients
    var dy_dtheta : Matrix? = null
    var dtheta_dW : Matrix? = null
    var dtheta_dX : Matrix? = null

    fun forward( inputs : Matrix ) : Matrix {
        X = inputs
        // Check is bias is required
        if ( requiresBias ) {
            theta = MatrixOps.dot( inputs , W!! ) + B
        }
        else {
            theta = MatrixOps.dot( inputs , W!! )
        }
        // Call the activation function
        y = activation.call( theta!! )
        return y!!
    }

    // These methods are called by the back propagation algorithm.
    fun initWeights( inputDims : Int ){
        W = MatrixOps.uniform( inputDims , units )
    }
    fun computeGradients() {
        dy_dtheta = activation.gradient( theta!! )
        dtheta_dW = X
        dtheta_dX = W
    }

}

