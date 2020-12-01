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

import kotlin.math.exp

class ActivationOps {

    class Sigmoid : Activation() {

        override fun call(x: Matrix) : Matrix {
            val activation = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    activation.set( i , j , sigmoid_( x.get( i , j ) ) )
                }
            }
            return activation
        }

        override fun gradient(x: Matrix): Matrix {
            val gradient = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    gradient.set( i , j , sigmoidGradient_( x.get( i , j ) ) )
                }
            }
            return gradient
        }

        private fun sigmoid_( x : Double ) : Double {
            return 1.0 / ( 1.0 + exp( -x ) )
        }
        private fun sigmoidGradient_( x : Double ) : Double {
            return sigmoid_(x) * ( 1.0 - sigmoid_(x) )
        }

    }

    class Softmax : Activation() {

        override fun call(x: Matrix): Matrix {
            val e_x = MatrixOps.exp( x - MatrixOps.max_along_axis0( x ) )
            return e_x * ( 1 / MatrixOps.sum_along_axis0( e_x ) )
        }

        override fun gradient(x: Matrix): Matrix {
            val e_x = MatrixOps.exp( x - MatrixOps.max_along_axis0( x ) )
            val s = e_x * ( 1 / MatrixOps.sum_along_axis0( e_x ) )
            return (s * ( s - 1.0 )) * -1.0
        }

    }

    class ReLU : Activation() {

        override fun call(x: Matrix): Matrix {
            val activation = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    activation.set( i , j , relu_( x.get( i , j ) ) )
                }
            }
            return activation
        }

        override fun gradient(x: Matrix): Matrix {
            val gradient = Matrix( x.m , x.n )
            for ( i in 0 until x.m ) {
                for ( j in 0 until x.n ) {
                    gradient.set( i , j , reluGradient_( x.get( i , j ) ) )
                }
            }
            return gradient
        }

        private fun relu_( x : Double ) : Double {
            return if ( x > 0.0 ){ x } else { 0.0 }
        }
        private fun reluGradient_( x : Double ) : Double {
            return if ( x > 0.0 ){ 1.0 } else { 0.0 }
        }

    }

}