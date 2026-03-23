using System;
using System.Collections.Generic;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using SequenceProcessing.Functions;
using SequenceProcessing.Parameters;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Classification
{
    [Serializable]
    public class GatedRecurrentUnitModel : RecurrentNeuralNetworkModel
    {
        /**
         * <summary>Creates a gated recurrent unit model with the given parameters and word embedding length.</summary>
         *
         * <param name="parameter">The neural network parameters.</param>
         * <param name="wordEmbeddingLength">The word embedding length.</param>
         */
        public GatedRecurrentUnitModel(ComputationalGraph.NeuralNetworkParameter parameter, int wordEmbeddingLength)
            : base(parameter, wordEmbeddingLength)
        {
            Switches = new List<Switch>();
        }

        /**
         * <summary>Trains the gated recurrent unit model with the given training set.</summary>
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
                for (var j = 0; j < 3; j++)
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
            var outputNodes = new List<ComputationalNode>();

            for (var k = 0; k < timeStep; k++)
            {
                Switches.Add(new Switch());

                var newOldLayers = new List<ComputationalNode>();
                var input = new MultiplicationNode(false, true);
                InputNodes.Add(input);

                ComputationalNode current = input;

                for (var i = 0; i < ((RecurrentNeuralNetworkParameter)Parameters).Size(); i++)
                {
                    ComputationalNode weightedNode;
                    ComputationalNode activationNode;

                    if (currentOldLayers.Count > 0)
                    {
                        weightedNode = this.AddEdge(current, weights[i * 3]);

                        var oldWithoutBias = this.AddEdge(currentOldLayers[i], new RemoveBias());
                        var recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i * 3]);
                        var sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);

                        var updateGate = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i * 2));

                        weightedNode = this.AddEdge(current, weights[(i * 3) + 1]);
                        recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[(i * 3) + 1]);
                        sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);

                        var resetGate = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction((i * 2) + 1));

                        weightedNode = this.AddEdge(current, weights[(i * 3) + 2]);
                        var resetHidden = this.AddEdge(resetGate, oldWithoutBias, false, true);
                        recurrentNode = this.AddEdge(resetHidden, recurrentWeights[(i * 3) + 2]);
                        sumNode = this.AddAdditionEdge(weightedNode, recurrentNode, false);

                        var candidateHidden = this.AddEdge(sumNode, new Tanh());
                        var negativeUpdateGate = this.AddEdge(updateGate, new Negation());
                        var oneMinusUpdateGate = this.AddEdge(negativeUpdateGate, new AdditionByConstant(1.0));

                        weightedNode = this.AddEdge(oneMinusUpdateGate, oldWithoutBias, false, true);
                        recurrentNode = this.AddEdge(candidateHidden, updateGate, false, true);
                        activationNode = this.AddAdditionEdge(weightedNode, recurrentNode, true);
                    }
                    else
                    {
                        weightedNode = this.AddEdge(current, weights[i * 3]);
                        var updateGate = this.AddEdge(
                            weightedNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i * 2));

                        weightedNode = this.AddEdge(current, weights[(i * 3) + 2]);
                        var candidateHidden = this.AddEdge(weightedNode, new Tanh());

                        activationNode = this.AddEdge(updateGate, candidateHidden, true, true);
                    }

                    current = activationNode;
                    newOldLayers.Add(activationNode);
                }

                currentOldLayers = newOldLayers;

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