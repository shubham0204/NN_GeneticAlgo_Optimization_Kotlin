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


import java.util.*
import kotlin.collections.ArrayList
import kotlin.collections.HashMap

// Class which uses the Genetic algorithm for optimization.
class Optimizer(
    private var paramChoices : HashMap<String,FloatArray> ,
    private var mutateProb : Float = 0.2f ,
    private var randomSelect : Float = 0.3f ,
    private var retain : Float = 0.3f
) {

    companion object {

        // Given a Network object, return the fitness score
        fun getFitnessScore( network: Network ) : Float {
            return network.score
        }

    }

    // Create an initial population from where the optimization can start, given the no. of
    // members/individuals required.
    fun createPopulation( numIndividuals : Int ) : ArrayList<Network> {
        val population = ArrayList<Network>()
        // Add each member
        for (i in 0 until numIndividuals) {
            val individual = Network()
            // Each member will have the same set of choices
            individual.paramChoices = paramChoices
            // Initialize this member with random picked choices
            individual.initializeNNWithRandomParams()
            population.add( individual )
        }
        return population
    }

    // Produce two children from their parents
    private fun breed( mother : Network , father : Network ) : ArrayList<Network> {
        val children = ArrayList<Network>()
        for( i in 0 until 2 ) {
            // Create parameters for the child NN
            val childParams = HashMap<String,Float>()
            for ( name in paramChoices.keys ){
                childParams[ name ] = arrayOf(
                        mother.networkParams[ name ] , father.networkParams[ name ] ).random() as Float
            }
            var child = Network()
            child.networkParams = childParams
            // Perform mutation
            if ( mutateProb > Random().nextFloat() ) {
                child = mutate( child )
            }
            children.add( child )
        }
        return children
    }

    // Mutate the parameters of the child NN
    private fun mutate( child : Network ) : Network {
        // Choose any random parameter which will be mutated
        val randomParam = paramChoices.keys.random()
        // Choose any random value for that parameter ( chosen above )
        child.networkParams[ randomParam ] = paramChoices[ randomParam ]!!.random()
        return child
    }

    fun getAvgScore(population : ArrayList<Network> ) : Float {
        val summation = population.map { network -> network.score }.toFloatArray().sum()
        return summation / population.size
    }

    fun evolve( currentPopulation : ArrayList<Network> ) : ArrayList<Network> {
        // Get the fitness scores of all individuals in this population
        var fitnessScores = currentPopulation.map{ individual -> getFitnessScore( individual ) }
        // Sort them in descending order
        fitnessScores = fitnessScores.sortedDescending()
        // Select top K individuals ( which have the highest fitness score ) from this population
        val retainLength = ( currentPopulation.size * retain ).toInt()
        val parents = ArrayList( currentPopulation.slice( 0..retainLength ) )
        // Add some more individuals ( the ones filtered above )
        for ( unwantedParent in currentPopulation.subList( retainLength , currentPopulation.size ) ) {
            if ( randomSelect > Random().nextFloat() ) {
                parents.add( unwantedParent )
            }
        }

        val parentsLength = parents.size
        val desiredLength = currentPopulation.size - parentsLength
        val children = ArrayList<Network>()

        while( children.size < desiredLength ) {

            val male = Random().nextInt( parents.size )
            val female = Random().nextInt( parents.size )

            if ( male != female ){
                val maleParent = parents[ male ]
                val femaleParent = parents[ female ]
                val babies = breed( maleParent , femaleParent )
                for ( baby in babies ){
                    children.add( baby )
                    println( "Adding children")
                }
            }
        }

        println( "Loop completed" )

        println( "currentPopulatin ${currentPopulation.size} " )
        println( "parents ${parents.size } " )
        println( "children ${children.size}")
        parents.addAll( children )
        return parents
    }


}
