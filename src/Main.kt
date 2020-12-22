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

fun main() {

    // Number of generations to be evolved
    val numGenerations = 10
    // Number of individuals in each population. In our case, this is the number of networks
    // which will be trained for 1 iteration so as to get the loss incurred by each NN.
    val population = 20

    val paramChoices = HashMap<String,FloatArray>()
    paramChoices[ "numLayers" ] = floatArrayOf( 2f , 3f , 4f )
    paramChoices[ "numNeurons" ] = floatArrayOf( 12f , 10f , 8f )
    paramChoices[ "learningRates" ] = floatArrayOf( 0.01f , 0.001f , 0.005f )

    // Create an optimizer.
    val optimizer = Optimizer( paramChoices )
    // Create a population with NNs consisting of randomly chosen parameters.
    var individuals = optimizer.createPopulation( population )

    for ( i in 0 until numGenerations ){
        println( "Evolve $i")
        for ( individual in individuals ){
            println( "Training an individual" )
            individual.train()
        }
        val avgScore = optimizer.getAvgScore( individuals )
        if ( i != numGenerations - 1 ){
            println( "Evolving this")
            individuals = optimizer.evolve( individuals )
        }

        println( "Evolution of $i done")


    }

    individuals.sortBy{ network -> network.score }
    individuals.reverse()

    val network = individuals[ 0 ]
    println( network.toString() )

}