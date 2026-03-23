using System;
using System.Collections.Generic;
using Classification.Performance;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using SequenceProcessing.Functions;
using SequenceProcessing.Parameters;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Classification
{
    [Serializable]
    public class RecurrentNeuralNetworkModel : ComputationalGraph.ComputationalGraph
    {
        protected readonly int WordEmbeddingLength;
        protected List<Switch> Switches;

        /**
         * <summary>Creates a recurrent neural network model with the given parameters and word embedding length.</summary>
         *
         * <param name="parameter">The neural network parameters.</param>
         * <param name="wordEmbeddingLength">The word embedding length.</param>
         */
        public RecurrentNeuralNetworkModel(NeuralNetworkParameter parameter, int wordEmbeddingLength)
            : base(parameter)
        {
            WordEmbeddingLength = wordEmbeddingLength;
            Switches = new List<Switch>();
        }

        /**
         * <summary>Creates input tensors and returns class labels for the given instance.</summary>
         *
         * <param name="instance">The input tensor instance.</param>
         * <returns>The class labels extracted from the instance.</returns>
         */
        protected List<int> CreateInputTensors(Tensor instance)
        {
            var classLabels = new List<int>();
            var timeStep = instance.GetShape()[0] / (WordEmbeddingLength + 1);
            var j = 0;

            for (var i = 0; i < this.InputNodes.Count - 1; i++)
            {
                if (i < timeStep)
                {
                    this.Switches[i].SetTurn(true);

                    var values = new List<double>();
                    for (var k = 0; k < WordEmbeddingLength; k++)
                    {
                        values.Add(instance.GetValue(new[] { j }));
                        j++;
                    }

                    classLabels.Add((int)instance.GetValue(new[] { j }));
                    j++;

                    InputNodes[i].SetValue(new Tensor(values, new[] { 1, values.Count }));
                }
                else
                {
                    this.Switches[i].SetTurn(false);

                    var values = new List<double>();
                    for (var k = 0; k < WordEmbeddingLength; k++)
                    {
                        values.Add(0.0);
                        j++;
                    }

                    classLabels.Add(0);
                    j++;

                    InputNodes[i].SetValue(new Tensor(values, new[] { 1, values.Count }));
                }
            }

            return classLabels;
        }

        /**
         * <summary>Trains the model internally with the given training set and randomizer.</summary>
         *
         * <param name="trainSet">The training set.</param>
         * <param name="random">The random generator used for shuffling.</param>
         */
        protected void Train(List<Tensor> trainSet, Random random)
        {
            for (var i = 0; i < Parameters.GetEpoch(); i++)
            {
                for (var j = 0; j < trainSet.Count; j++)
                {
                    var i1 = random.Next(trainSet.Count);
                    var i2 = random.Next(trainSet.Count);

                    var tmp = trainSet[i1];
                    trainSet[i1] = trainSet[i2];
                    trainSet[i2] = tmp;
                }

                foreach (var instance in trainSet)
                {
                    var classLabels = CreateInputTensors(instance);
                    var classLabelValues = new List<double>();

                    foreach (var classLabel in classLabels)
                    {
                        for (var inputIndex = 0; inputIndex < ((RecurrentNeuralNetworkParameter)this.Parameters).GetClassLabelSize(); inputIndex++)
                        {
                            if (inputIndex == classLabel)
                            {
                                classLabelValues.Add(1.0);
                            }
                            else
                            {
                                classLabelValues.Add(0.0);
                            }
                        }
                    }

                    InputNodes[this.InputNodes.Count - 1].SetValue(
                        new Tensor(
                            classLabelValues,
                            new[]
                            {
                                classLabels.Count,
                                ((RecurrentNeuralNetworkParameter)this.Parameters).GetClassLabelSize()
                            }
                        )
                    );

                    this.ForwardCalculation();
                    this.Backpropagation();
                }

                Parameters.GetOptimizer().SetLearningRate();
            }
        }

        /**
         * <summary>Finds the maximum time step in the given training set.</summary>
         *
         * <param name="trainSet">The training set.</param>
         * <returns>The maximum time step.</returns>
         */
        protected int FindTimeStep(List<Tensor> trainSet)
        {
            var timeStep = -1;

            foreach (var tensor in trainSet)
            {
                var size = tensor.GetShape()[0];
                if (timeStep < size / (WordEmbeddingLength + 1))
                {
                    timeStep = size / (WordEmbeddingLength + 1);
                }
            }

            return timeStep;
        }

        /**
         * <summary>Trains the many-to-many recurrent neural network model.</summary>
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
                this.Switches.Add(new Switch());

                var newOldLayers = new List<ComputationalNode>();
                var input = new MultiplicationNode(false, true);
                InputNodes.Add(input);

                ComputationalNode current = input;

                for (var i = 0; i < ((RecurrentNeuralNetworkParameter)Parameters).Size(); i++)
                {
                    ComputationalNode aw;
                    ComputationalNode activationNode;

                    if (currentOldLayers.Count > 0)
                    {
                        aw = this.AddEdge(current, weights[i]);

                        var oldWithoutBias = this.AddEdge(currentOldLayers[i], new RemoveBias());
                        var recurrentNode = this.AddEdge(oldWithoutBias, recurrentWeights[i]);
                        var sumNode = this.AddAdditionEdge(aw, recurrentNode, false);

                        activationNode = this.AddEdge(
                            sumNode,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i),
                            true);
                    }
                    else
                    {
                        aw = this.AddEdge(current, weights[i], false);

                        activationNode = this.AddEdge(
                            aw,
                            ((RecurrentNeuralNetworkParameter)Parameters).GetActivationFunction(i),
                            true);
                    }

                    current = activationNode;
                    newOldLayers.Add(activationNode);
                }

                currentOldLayers = newOldLayers;

                var node = this.AddEdge(current, weights[weights.Count - 1]);
                outputNodes.Add(this.AddEdge(node, Switches[k]));
            }

            var concatenatedNode = (ConcatenatedNode)this.ConcatEdges(outputNodes, 0);
            this.OutputNode = this.AddEdge(concatenatedNode, new Softmax());

            var classLabelNode = new ComputationalNode(false, false);
            this.InputNodes.Add(classLabelNode);

            var lossInputs = new List<ComputationalNode>();
            lossInputs.Add(this.OutputNode);
            lossInputs.Add(classLabelNode);

            this.AddFunctionEdge(lossInputs, Parameters.GetLossFunction(), false);
            Train(trainSet, random);
        }

        /**
         * <summary>Returns the predicted output values from the output node.</summary>
         *
         * <param name="outputNode">The output node.</param>
         * <returns>The predicted class indices.</returns>
         */
        protected override List<double> GetOutputValue(ComputationalNode outputNode)
        {
            var classLabels = new List<double>();

            for (var i = 0; i < outputNode.GetValue().GetShape()[0]; i++)
            {
                var index = -1;
                var max = double.MinValue;

                for (var j = 0; j < outputNode.GetValue().GetShape()[1]; j++)
                {
                    if (max < outputNode.GetValue().GetValue(new[] { i, j }))
                    {
                        max = outputNode.GetValue().GetValue(new[] { i, j });
                        index = j;
                    }
                }

                classLabels.Add(index);
            }

            return classLabels;
        }

        /**
         * <summary>Tests the model with the given test set.</summary>
         *
         * <param name="testSet">The test set.</param>
         * <returns>The classification performance.</returns>
         */
        public override ClassificationPerformance Test(List<Tensor> testSet)
        {
            var count = 0;
            var total = 0;

            foreach (var instance in testSet)
            {
                var goldClassLabels = CreateInputTensors(instance);
                var classLabels = this.Predict();

                for (var j = 0; j < (instance.GetShape()[0] / (WordEmbeddingLength + 1)); j++)
                {
                    if (goldClassLabels[j] == (int)classLabels[j])
                    {
                        count++;
                    }

                    total++;
                }
            }

            return new ClassificationPerformance((count + 0.0) / total);
        }
    }
}