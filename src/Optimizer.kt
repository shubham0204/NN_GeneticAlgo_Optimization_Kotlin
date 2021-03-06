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
            for ( param in paramChoices.keys ){
                childParams[ param ] = arrayOf(
                        mother.networkParams[ param ] , father.networkParams[ param ] ).random() as Float
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

    // Get the average fitness score of all members in the given population.
    fun getAvgScore(population : ArrayList<Network> ) : Float {
        val summation = population.map { network -> network.score }.toFloatArray().sum()
        return summation / population.size
    }

    fun evolve( currentPopulation : ArrayList<Network> ) : ArrayList<Network> {
        // Get the fitness scores of all individuals in this population and sort them.
        val populationSorted = currentPopulation.sortedWith( kotlin.Comparator {
            o1, o2 -> o2.score.compareTo( o1.score )  })
        println( "Initial population size ${populationSorted.size}")
        // Select top K individuals ( which have the highest fitness score ) from this population
        val retainLength = ( currentPopulation.size * retain ).toInt()
        val parents = ArrayList( populationSorted.slice( 0..retainLength ) )
        println( "Population size after selecting K individuals ${parents.size}")
        // Add some more individuals ( the ones filtered above )
        for ( unwantedParent in currentPopulation.subList( retainLength , currentPopulation.size ) ) {
            if ( randomSelect > Random().nextFloat() ) {
                parents.add( unwantedParent )
            }
        }
        println( "Population size after adding some more parents ${parents.size}")

        // Create an array ( of length (total_individuals - selected_parents) ) to store children.
        val parentsLength = parents.size
        val desiredLength = currentPopulation.size - parentsLength
        val children = ArrayList<Network>()

        println( "Desired length ${desiredLength}")

        while( children.size < desiredLength ) {

            // Select any two parents
            val male = Random().nextInt( parents.size )
            val female = Random().nextInt( parents.size )

            // Check if they are different from each other
            if ( male != female ){
                val maleParent = parents[ male ]
                val femaleParent = parents[ female ]
                // breeding
                val babies = breed( maleParent , femaleParent )
                // Append the babies to the arraylist
                children.addAll( babies )
            }

        }

        println( "children Size ${children.size} " )

  /*      println( "retainLength Size ${desiredLength}} " )
        println( "children ${children.size}")*/

        parents.addAll( children )

        println( "new generation ${parents.size } " )
        return parents
    }


}
