<template>
    <div id="app">
        <input type="range" min="21" max="25" step="0.01" v-model="massExp">
        <span>{{mass.toExponential(2)}}</span>
        <br>

        <input type="range" min="10" max="100" step="1" v-model="gammaPercent">
        <span>{{gamma}}</span>


        <br>
        <br>

        <h1>{{centerResult}}</h1>

        <canvas ref="test"></canvas>
    </div>
</template>

<script>
    import * as tf from '@tensorflow/tfjs';

    // tf.enableDebugMode();

    const scaler = {
        "means": [29.265522791396066, 2.670982382745096, 7.513687825266656e+23, 0.5083343801015716, 0.14662571984313735, 0.1523470950588237],
        "stds": [20.60722972921698, 1.3161197180526485, 1.4839680415661546e+24, 0.3264606872642621, 0.039902761966033776, 0.039545391017794064]
    }; //TODO: load from json

    const res = 100;
    const pixels = res * res;
    const means = tf.tensor(scaler.means).tile([pixels]).reshape([pixels, 6]);
    const stds = tf.tensor(scaler.stds).tile([pixels]).reshape([pixels, 6]);
    export default {
        name: 'app',
        data() {
            return {
                massExp: 23,
                calculating: false,
                result: 0,
                gammaPercent: 21
            }
        },
        watch: {
            parameter_list() {
                if (this.calculating) {
                    console.info("abort");
                    return false;
                }
                if (!this.model) {
                    console.warn("model not yet loaded");
                    return false;
                }
                this.calculating = true;
                const tensor = tf.tensor(this.parameter_list);
                const scaled_tensor = tensor.subStrict(means).divStrict(stds);
                tensor.dispose();

                const resultTensor = this.model.predictOnBatch(scaled_tensor, {verbose: true});
                scaled_tensor.dispose();
                resultTensor.data().then(numberlist => {
                    this.result = numberlist;
                    const context = this.$refs.test.getContext('2d');
                    const imagedata = [];
                    for (let i = 0; i < numberlist.length; i++) {
                        const color = 255 - Math.round(numberlist[i] * 255);
                        imagedata.push(color, color, color, 255)
                    }
                    const pixeldata = new Uint8ClampedArray(imagedata);
                    context.canvas.width = res;
                    context.canvas.height = res;
                    context.putImageData(new ImageData(pixeldata, res, res), 0, 0);
                    resultTensor.dispose();
                    this.calculating = false;
                });
            }
        },
        computed: {
            mass() {
                return Math.pow(10, this.massExp);
            },
            gamma() {
                return this.gammaPercent / 100;
            },
            centerResult() {
                return this.result[(res * res + res) / 2]
            },
            parameter_list() {
                const v = 2.31;
                const gamma = 0.21;
                const target_water_fraction = 0.15;
                const projectile_water_fraction = 0.15;
                const datalist = [];

                for (let j = 0; j < res; j++) {
                    for (let i = 0; i < res; i++) {

                        const entry = [
                            i / res * 60,
                            j / res * 5.5,
                            this.mass,
                            this.gamma,
                            target_water_fraction,
                            projectile_water_fraction
                        ];
                        datalist.push(entry)
                    }
                }
                return datalist
            }
        },
        mounted() {
            tf.loadLayersModel('/models/model.json').then(model => {
                this.model = model;
            });
        }
    }
</script>

<style>
    #app {
        font-family: 'Avenir', Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-align: center;
        color: #2c3e50;
        margin-top: 60px;
    }

    canvas {
        width: 100%;
        image-rendering: pixelated;
        transform: scaleY(-1);
    }
</style>
