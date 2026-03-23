using System;
using System.Collections.Generic;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using SequenceProcessing.Functions;
using SequenceProcessing.Parameters;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Classification
{
    [Serializable]
    public class LongShortTermMemoryModel : RecurrentNeuralNetworkModel
    {
        /**
         * <summary>Creates a long short-term memory model with the given parameters and word embedding length.</summary>
         *
         * <param name="parameter">The neural network parameters.</param>
         * <param name="wordEmbeddingLength">The word embedding length.</param>
         */
        public LongShortTermMemoryModel(NeuralNetworkParameter parameter, int wordEmbeddingLength)
            : base(parameter, wordEmbeddingLength)
        {
            Switches = new List<Switch>();
        }

        /**
         * <summary>Trains the long short-term memory model with the given training set.</summary>
         *
         * <param name="trainSet">The training set.</param>
         */
        public override void Train(List<Tensor> trainSet)
        {
            var random = new Random(Parameters.GetSeed());
            var timeStep = FindTimeStep(trainSet);

            var weights = new List<ComputationalNode>();
            var recurrentWeights = new List<ComputationalNode>();

            var currentLength = WordEmbeddingLength + 1;

            for (var i = 0; i < ((RecurrentNeuralNetworkParameter)Parameters).Size(); i++)
            {
                for (var j = 0; j < 4; j++)
                {
                    weights.Add(
                        new MultiplicationNode(
                            new Tensor(
                                Parameters.InitializeWeights(
                                    currentLength,
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i),
                                    random),
                                new[]
                                {
                                    currentLength,
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i)
                                }
                            )
                        )
                    );

                    recurrentWeights.Add(
                        new MultiplicationNode(
                            new Tensor(
                                Parameters.InitializeWeights(
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i),
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i),
                                    random),
                                new[]
                                {
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i),
                                    ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i)
                                }
                            )
                        )
                    );
                }

                currentLength = ((RecurrentNeuralNetworkParameter)Parameters).GetHiddenLayer(i) + 1;
            }

            weights.Add(
                new MultiplicationNode(
                    new Tensor(
                        Parameters.InitializeWeights(
                            currentLength,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetClassLabelSize(),
                            random),
                        new[]
                        {
                            currentLength,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetClassLabelSize()
                        }
                    )
                )
            );

            var currentOldLayers = new List<ComputationalNode>();
            var currentOldCValues = new List<ComputationalNode>();
            var outputNodes = new List<ComputationalNode>();

            for (var k = 0; k < timeStep; k++)
            {
                Switches.Add(new Switch());

                var newOldLayers = new List<ComputationalNode>();
                var newOldCValues = new List<ComputationalNode>();

                var input = new MultiplicationNode(false, true);
                InputNodes.Add(input);

                ComputationalNode current = input;

                for (var i = 0; i < weights.Count - 1; i += 4)
                {
                    ComputationalNode weightedNode;
                    ComputationalNode activationNode;
                    ComputationalNode cellStateNode;

                    if (currentOldLayers.Count > 0)
                    {
                        weightedNode = this.AddEdge(current, weights[i]);

                        var oldWithoutBias = this.AddEdge(currentOldLayers[i / 4], new RemoveBias());

                        var recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i]);
                        var sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);
                        var inputGate = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i));

                        weightedNode = this.AddEdge(current, weights[i + 1]);
                        recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i + 1]);
                        sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);
                        var forgetGate = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i + 1));

                        weightedNode = this.AddEdge(current, weights[i + 2]);
                        recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i + 2]);
                        sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);
                        var outputGate = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i + 2));

                        weightedNode = this.AddEdge(current, weights[i + 3]);
                        recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i + 3]);
                        sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);
                        var candidateCell = this.AddEdge(sumNode, new Tanh());

                        var forgetCellProduct = this.AddEdge(forgetGate, currentOldCValues[i / 4], false, true);
                        var inputCellProduct = this.AddEdge(inputGate, candidateCell, false, true);
                        var combinedCell = this.AddAdditionEdge(forgetCellProduct, inputCellProduct, false);

                        cellStateNode = this.AddEdge(
                            combinedCell,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i + 3));

                        var tanhCellState = this.AddEdge(cellStateNode, new Tanh());
                        activationNode = this.AddEdge(tanhCellState, outputGate, true, true);
                    }
                    else
                    {
                        weightedNode = this.AddEdge(current, weights[i]);
                        var inputGate = this.AddEdge(
                            weightedNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i));

                        weightedNode = this.AddEdge(current, weights[i + 1]);
                        var outputGate = this.AddEdge(
                            weightedNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i + 2));

                        weightedNode = this.AddEdge(current, weights[i + 3]);
                        var candidateCell = this.AddEdge(weightedNode, new Tanh());

                        var inputCellProduct = this.AddEdge(inputGate, candidateCell, false, true);

                        cellStateNode = this.AddEdge(
                            inputCellProduct,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i + 3));

                        var tanhCellState = this.AddEdge(cellStateNode, new Tanh());
                        activationNode = this.AddEdge(tanhCellState, outputGate, true, true);
                    }

                    current = activationNode;
                    newOldLayers.Add(activationNode);
                    newOldCValues.Add(cellStateNode);
                }

                currentOldLayers = newOldLayers;
                currentOldCValues = newOldCValues;

                var node = this.AddEdge(current, weights[weights.Count - 1]);
                outputNodes.Add(this.AddEdge(node, Switches[k]));
            }

            var concatenatedNode = (ConcatenatedNode)this.ConcatEdges(outputNodes, 0);
            OutputNode = this.AddEdge(concatenatedNode, new Softmax());

            var classLabelNode = new ComputationalNode(false, false);
            InputNodes.Add(classLabelNode);

            var lossInputs = new List<ComputationalNode>();
            lossInputs.Add(OutputNode);
            lossInputs.Add(classLabelNode);

            this.AddFunctionEdge(lossInputs, Parameters.GetLossFunction(), false);
            Train(trainSet, random);
        }
    }
}