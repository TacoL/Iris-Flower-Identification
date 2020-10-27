using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Iris_Flower_Identification
{
    public class NeuralNetwork
    {
        //hyperparameters
        public static int numSamples = 150;
        public static int numInputs = 4;
        public static int[] hiddenLayers = { 7 }; //best values are 7, 8, 9
        public static int numOutputs = 3;
        public static int numEpochs = 5000;
        public static double learningRate = .22;
        public static double momentumScalar = .12;

        //references
        double[][] inputs;
        double[][] targets;

        List<Matrix> layers = new List<Matrix>();
        List<Matrix> weights = new List<Matrix>();
        List<Matrix> biases = new List<Matrix>();
        int epochNum = 0;
        public static List<Matrix> previousWeightGradients;
        public static List<Matrix> previousBiasGradients;

        public NeuralNetwork()
        {
            GenerateData(out inputs, out targets);
            Generate();

            for (int i = 0; i < numEpochs; i++)
            {
                Console.WriteLine("Epoch: " + (i + 1));
                backPropagate();
                epochNum++;
            }

            StreamReader sr = new StreamReader(File.OpenRead(@"IRIS.csv"));
            String line = sr.ReadLine(); //skips first line
            while ((line = sr.ReadLine()) != null)
            {
                String[] dividedString = line.Split(',');
                Matrix results = forwardPass(new double[] { Double.Parse(dividedString[0]), Double.Parse(dividedString[1]), Double.Parse(dividedString[2]), Double.Parse(dividedString[3]) });

                int rowWithMaxValue = 0;
                for (int row = 0; row < results.numRows; row++)
                {
                    if (results.matrix[row, 0] > results.matrix[rowWithMaxValue, 0])
                    {
                        rowWithMaxValue = row;
                    }
                }

                String[] names = { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
                if (names[rowWithMaxValue] == dividedString[4])
                {
                    Console.WriteLine("Match: " + names[rowWithMaxValue]);
                }
                else
                {
                    Console.WriteLine("Error: Predicted = " + names[rowWithMaxValue] + ", Actual = " + dividedString[4]);
                }
            }
        }

        public void GenerateData(out double[][] inputs, out double[][] targets)
        {
            inputs = new double[numSamples][];
            targets = new double[numSamples][];

            int sampleIdx = 0;

            StreamReader sr = new StreamReader(File.OpenRead(@"IRIS.csv"));
            String line = sr.ReadLine(); //skips first line
            while ((line = sr.ReadLine()) != null)
            {
                inputs[sampleIdx] = new double[numInputs];
                targets[sampleIdx] = new double[numOutputs];
                String[] dividedString = line.Split(',');
                for (int i = 0; i < dividedString.Length; i++)
                {
                    if (i < 4) //inputs
                    {
                        double value = Double.Parse(dividedString[i]);
                        inputs[sampleIdx][i] = value;
                    }
                    else //output
                    {
                        switch (dividedString[i])
                        {
                            case "Iris-setosa":
                                targets[sampleIdx][0] = 1;
                                targets[sampleIdx][1] = 0;
                                targets[sampleIdx][2] = 0;
                                break;
                            case "Iris-versicolor":
                                targets[sampleIdx][0] = 0;
                                targets[sampleIdx][1] = 1;
                                targets[sampleIdx][2] = 0;
                                break;
                            case "Iris-virginica":
                                targets[sampleIdx][0] = 0;
                                targets[sampleIdx][1] = 0;
                                targets[sampleIdx][2] = 1;
                                break;
                        }
                    }
                }
                sampleIdx++;
            }
        }

        public void Generate()
        {
            layers.Add(new Matrix(numInputs, 1));
            foreach (int i in hiddenLayers)
            {
                layers.Add(new Matrix(i, 1));
            }
            layers.Add(new Matrix(numOutputs, 1));
            layers.Add(new Matrix(numOutputs, 1)); //for the softmax layer

            for (int i = 0; i < layers.Count; i++)
            {
                if (i == 0 || i == layers.Count - 1)  //don't add weights or biases for the softmax layer
                {
                    weights.Add(new Matrix(0, 0));
                    biases.Add(new Matrix(0, 0));
                }
                else
                {
                    weights.Add(Matrix.Randomize(new Matrix(layers[i].numRows, layers[i - 1].numRows)));
                    biases.Add(Matrix.Randomize(new Matrix(layers[i].numRows, 1)));
                }
            }
        }

        public Matrix forwardPass(double[] inputs)
        {
            layers[0] = new Matrix(inputs);
            for (int layerIdx = 1; layerIdx < layers.Count; layerIdx++) //skip the input layer
            {
                Matrix previousLayer = layers[layerIdx - 1];
                if (layerIdx != layers.Count - 1)
                {
                    Matrix weightMatrix = weights[layerIdx];
                    Matrix biasMatrix = biases[layerIdx];

                    Matrix outputBeforeActivation = Matrix.Addition(Matrix.Multiply(weightMatrix, previousLayer), biasMatrix);
                    layers[layerIdx] = Matrix.ApplyActivation(outputBeforeActivation);
                }
                else
                {
                    layers[layerIdx] = Matrix.ApplySoftmax(previousLayer);
                }
            }

            return layers[layers.Count - 1]; //returns the probability vector
        }

        public void backPropagate()
        {
            List<Matrix> totalWeightGradients = new List<Matrix>(layers.Count);
            List<Matrix> totalBiasGradients = new List<Matrix>(layers.Count);

            for (int i = 0; i < layers.Count; i++)
            {
                totalWeightGradients.Add(new Matrix(0, 0));
                totalBiasGradients.Add(new Matrix(0, 0));
            }

            //get all the weight gradients
            for (int sampleIdx = 0; sampleIdx < numSamples; sampleIdx++)
            {
                double[] sampleInput = inputs[sampleIdx];
                Matrix outputs = forwardPass(sampleInput);
                Matrix targetsMatrix = new Matrix(targets[sampleIdx]);
                Matrix errorMatrix = Matrix.Subtract(outputs, targetsMatrix);

                List<Matrix> activationGradientsMatrix = new List<Matrix>();
                for (int i = 0; i < layers.Count; i++)
                {
                    activationGradientsMatrix.Add(new Matrix(0, 0));
                }

                for (int layerIdx = layers.Count - 1; layerIdx > 0; layerIdx--)
                {
                    Matrix currentLayer = layers[layerIdx];
                    Matrix previousLayer = layers[layerIdx - 1];
                    Matrix weightMatrix = weights[layerIdx];
                    Matrix biasMatrix = biases[layerIdx];
                    Matrix activationGradient = new Matrix(currentLayer.numRows, 1);

                    Matrix outputBeforeDerivative = Matrix.Addition(Matrix.Multiply(weightMatrix, previousLayer), biasMatrix);

                    Matrix weightGradient = new Matrix(weightMatrix.numRows, weightMatrix.numColumns);
                    Matrix biasGradient = new Matrix(biasMatrix.numRows, biasMatrix.numColumns);

                    if (layerIdx == layers.Count - 1)
                    {
                        for (int row = 0; row < currentLayer.numRows; row++)
                        {
                            activationGradient.matrix[row, 0] = 2 * errorMatrix.matrix[row, 0];
                        }
                        activationGradientsMatrix[layerIdx] = activationGradient;
                    }

                    for (int row = 0; row < weightGradient.numRows; row++)
                    {
                        for (int column = 0; column < weightGradient.numColumns; column++)
                        {
                            double x = previousLayer.matrix[column, 0] * applyDerivativeActivation(outputBeforeDerivative.matrix[row, 0]);

                            if (layerIdx == layers.Count - 2) //output layer, before softamx
                            {
                                activationGradient.matrix[row, 0] = Matrix.ApplySoftmaxDerivative(currentLayer).matrix[row, 0] * activationGradientsMatrix[layerIdx + 1].matrix[row, 0];
                            }
                            else
                            {
                                Matrix layerInFront = layers[layerIdx + 1];
                                Matrix weightInFront = weights[layerIdx + 1];
                                Matrix biasInFront = biases[layerIdx + 1];
                                Matrix frontOutputBeforeActivation = Matrix.Addition(Matrix.Multiply(weightInFront, currentLayer), biasInFront);
                                double thisNeuronActivationGradient = 0;
                                for (int rowFront = 0; rowFront < layerInFront.numRows; rowFront++)
                                {
                                    thisNeuronActivationGradient += weightInFront.matrix[rowFront, row] * applyDerivativeActivation(frontOutputBeforeActivation.matrix[rowFront, 0]) * activationGradientsMatrix[layerIdx + 1].matrix[rowFront, 0];
                                }
                                activationGradient.matrix[row, 0] = thisNeuronActivationGradient;
                            }

                            weightGradient.matrix[row, column] = x * activationGradient.matrix[row, 0];
                            activationGradientsMatrix[layerIdx] = activationGradient;
                        }
                    }


                    for (int row = 0; row < biasGradient.numRows; row++)
                    {
                        if (layerIdx == layers.Count - 2)
                        {
                            biasGradient.matrix[row, 0] = applyDerivativeActivation(outputBeforeDerivative.matrix[row, 0]) * activationGradientsMatrix[layerIdx].matrix[row, 0];
                        }
                        else
                        {
                            Matrix layerInFront = layers[layerIdx + 1];
                            Matrix weightInFront = weights[layerIdx + 1];
                            Matrix biasInFront = biases[layerIdx + 1];
                            Matrix frontOutputBeforeDerivative = Matrix.Addition(Matrix.Multiply(weightInFront, currentLayer), biasInFront);
                            for (int rowFront = 0; rowFront < layerInFront.numRows; rowFront++)
                            {
                                double x = applyDerivativeActivation(frontOutputBeforeDerivative.matrix[rowFront, 0]) * activationGradientsMatrix[layerIdx + 1].matrix[rowFront, 0];
                                biasGradient.matrix[row, 0] += x * applyDerivativeActivation(outputBeforeDerivative.matrix[row, 0]) * weightInFront.matrix[rowFront, row];
                            }
                        }
                    }

                    if (sampleIdx == 0)
                    {
                        totalWeightGradients[layerIdx] = weightGradient;
                        totalBiasGradients[layerIdx] = biasGradient;
                    }
                    else
                    {
                        totalWeightGradients[layerIdx] = Matrix.Addition(totalWeightGradients[layerIdx], weightGradient);
                        totalBiasGradients[layerIdx] = Matrix.Addition(totalBiasGradients[layerIdx], biasGradient);
                    }
                }
            }

            for (int i = 1; i < totalWeightGradients.Count; i++)
            {
                Matrix.DivideByConstant(totalWeightGradients[i], numSamples);
                Matrix.DivideByConstant(totalBiasGradients[i], numSamples);
            }

            //update weights and biases
            for (int idx = 0; idx < layers.Count; idx++)
            {
                Matrix applyLRWeights = Matrix.DivideByConstant(totalWeightGradients[idx], 1 / learningRate);
                Matrix applyLRBiases = Matrix.DivideByConstant(totalBiasGradients[idx], 1 / learningRate);

                if (epochNum == 0)
                {
                    weights[idx] = Matrix.Subtract(weights[idx], applyLRWeights);
                    biases[idx] = Matrix.Subtract(biases[idx], applyLRBiases);
                }
                else
                {
                    List<Matrix> previousWeightGradients = getPWG();
                    List<Matrix> previousBiasGradients = getPBG();
                    weights[idx] = Matrix.Subtract(weights[idx], Matrix.Addition(applyLRWeights, Matrix.DivideByConstant(previousWeightGradients[idx], 1 / momentumScalar)));
                    biases[idx] = Matrix.Subtract(biases[idx], Matrix.Addition(applyLRBiases, Matrix.DivideByConstant(previousBiasGradients[idx], 1 / momentumScalar)));
                }
            }

            setPWG(totalWeightGradients);
            setPBG(totalBiasGradients);
        }

        public static void setPWG(List<Matrix> a)
        {
            previousWeightGradients = a;
        }

        public static List<Matrix> getPWG()
        {
            return previousWeightGradients;
        }

        public static void setPBG(List<Matrix> a)
        {
            previousBiasGradients = a;
        }

        public static List<Matrix> getPBG()
        {
            return previousBiasGradients;
        }

        public static double applyDerivativeActivation(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }
    }
}
