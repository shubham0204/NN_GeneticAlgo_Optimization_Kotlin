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

import NeuralNetwork.ActivationOps
import NeuralNetwork.Dense
import NeuralNetwork.MatrixOps
import NeuralNetwork.Model

// The class which trains the NN and stores the fitness score.
class Train {

    // NN Model
    private lateinit var model : Model

    // Inputs given to the NN
    private val inputs = MatrixOps.uniform( 1 , 3 )

    // Labels for the given input
    private val outputs = MatrixOps.uniform( 1 , 2 )

    // Number of iterations to train for
    private val numIterations : Int = 1

    // Input/Output dimensions
    private val inputDims : Int = 3
    private var outputDims : Int = 2

    // Compile the model with given hyperparameters.
    private fun compileModel( networkParams : HashMap<String,Float> ) {
        val numLayers = networkParams[ "numLayers" ]!!
        val numNeurons = networkParams[ "numNeurons" ]!!
        val learningRate = networkParams[ "learningRates" ]!!
        // Add required number layers and neurons in each
        val layers = ArrayList<Dense>()
        for ( i in 0 until numLayers.toInt() ) {
            layers.add( Dense( numNeurons.toInt() , ActivationOps.ReLU() , false ) )
        }
        layers.add( Dense( outputDims , ActivationOps.ReLU() , false ) )

        model = Model( inputDims , layers.toTypedArray() , learningRate.toDouble() )
        model.compile()
    }

    // Train a member ( neural network ) for a given number of epochs.
    // Return the fitness score which is ( 5 / loss )
    fun trainAndScore( networkParams : HashMap<String,Float> ) : Float {
        compileModel( networkParams )
        for (i in 0 until numIterations ) {
            println( "Training this individual for ${i+1} epochs")
            model.forward(inputs, outputs)
            model.backward()
        }
        val loss = MatrixOps.sum_along_axis0(outputs - model.forward(inputs, outputs)) / outputDims
        return ( 5.0 / loss).toFloat()
    }



}
