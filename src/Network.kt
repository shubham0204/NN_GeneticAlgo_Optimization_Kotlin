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

// Helper class for handling interactions with the Neural Network
class Network() {

    // The fitness score for this network
    var score = 0.0f

    // The network parameters ( hyperparameters ) for this NN
    var networkParams = HashMap<String,Float>()

    // Choices available for the hyperparameter
    var paramChoices = HashMap<String,FloatArray>()

    // NN trainer
    private var trainer : Train = Train()

    // Initialize this NN with random hyperparameters
    fun initializeNNWithRandomParams() {
        for ( key in paramChoices.keys ) {
            networkParams[ key ] = (paramChoices[ key ] as FloatArray).random()
        }
    }

    // Train this NN and store the fitness score
    fun train() {
        val score = trainer.trainAndScore( networkParams )
        this.score = score
    }

    override fun toString(): String {
        return networkParams.toString()
    }

}