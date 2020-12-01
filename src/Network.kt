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


class Network() {

    var score = 0.0f
    var networkParams = HashMap<String,Float>()
    var paramChoices = HashMap<String,FloatArray>()
    private var trainer : Train = Train()

    fun initializeNNWithRandomParams() {
        for ( key in paramChoices.keys ) {
            networkParams[ key ] = (paramChoices[ key ] as FloatArray).random()
        }
    }

    fun train() {
        val score = trainer.trainAndScore( networkParams )
        this.score = score

    }

}