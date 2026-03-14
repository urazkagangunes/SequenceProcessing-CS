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
        protected readonly int wordEmbeddingLength;
        protected List<Switch> switches;

        public RecurrentNeuralNetworkModel(NeuralNetworkParameter parameter, int wordEmbeddingLength)
            : base(parameter)
        {
            this.wordEmbeddingLength = wordEmbeddingLength;
            this.switches = new List<Switch>();
        }

        protected List<int> createInputTensors(Tensor instance)
        {
            List<int> classLabels = new List<int>();
            int timeStep = instance.GetShape()[0] / (wordEmbeddingLength + 1);
            int j = 0;

            for (int i = 0; i < this.inputNodes.Count - 1; i++)
            {
                if (i < timeStep)
                {
                    this.switches[i].setTurn(true);

                    List<double> values = new List<double>();
                    for (int k = 0; k < wordEmbeddingLength; k++)
                    {
                        values.Add(instance.GetValue(new int[] { j }));
                        j++;
                    }

                    classLabels.Add((int)instance.GetValue(new int[] { j }));
                    j++;

                    inputNodes[i].setValue(new Tensor(values, new int[] { 1, values.Count }));
                }
                else
                {
                    this.switches[i].setTurn(false);

                    List<double> values = new List<double>();
                    for (int k = 0; k < wordEmbeddingLength; k++)
                    {
                        values.Add(0.0);
                        j++;
                    }

                    classLabels.Add(0);
                    j++;

                    inputNodes[i].setValue(new Tensor(values, new int[] { 1, values.Count }));
                }
            }

            return classLabels;
        }

        protected void train(List<Tensor> trainSet, Random random)
        {
            for (int i = 0; i < parameters.getEpoch(); i++)
            {
                for (int j = 0; j < trainSet.Count; j++)
                {
                    int i1 = random.Next(trainSet.Count);
                    int i2 = random.Next(trainSet.Count);

                    Tensor tmp = trainSet[i1];
                    trainSet[i1] = trainSet[i2];
                    trainSet[i2] = tmp;
                }

                foreach (Tensor instance in trainSet)
                {
                    List<int> classLabels = createInputTensors(instance);
                    List<double> classLabelValues = new List<double>();

                    foreach (int classLabel in classLabels)
                    {
                        for (int inputIndex = 0; inputIndex < ((RecurrentNeuralNetworkParameter)this.parameters).getClassLabelSize(); inputIndex++)
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

                    inputNodes[this.inputNodes.Count - 1].setValue(
                        new Tensor(
                            classLabelValues,
                            new int[]
                            {
                                classLabels.Count,
                                ((RecurrentNeuralNetworkParameter)this.parameters).getClassLabelSize()
                            }
                        )
                    );

                    this.forwardCalculation();
                    this.backpropagation();
                }

                parameters.getOptimizer().setLearningRate();
            }
        }

        protected int findTimeStep(List<Tensor> trainSet)
        {
            int timeStep = -1;

            foreach (Tensor tensor in trainSet)
            {
                int size = tensor.GetShape()[0];
                if (timeStep < size / (wordEmbeddingLength + 1))
                {
                    timeStep = size / (wordEmbeddingLength + 1);
                }
            }

            return timeStep;
        }

        // Many-to-Many RNN
        public override void train(List<Tensor> trainSet)
        {
            Random random = new Random(parameters.GetSeed());
            int timeStep = findTimeStep(trainSet);

            List<ComputationalNode> weights = new List<ComputationalNode>();
            List<ComputationalNode> recurrentWeights = new List<ComputationalNode>();

            int currentLength = wordEmbeddingLength + 1;

            for (int i = 0; i < ((RecurrentNeuralNetworkParameter)parameters).size(); i++)
            {
                weights.Add(
                    new MultiplicationNode(
                        new Tensor(
                            parameters.initializeWeights(
                                currentLength,
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i),
                                random),
                            new int[]
                            {
                                currentLength,
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i)
                            }
                        )
                    )
                );

                recurrentWeights.Add(
                    new MultiplicationNode(
                        new Tensor(
                            parameters.initializeWeights(
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i),
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i),
                                random),
                            new int[]
                            {
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i),
                                ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i)
                            }
                        )
                    )
                );

                currentLength = ((RecurrentNeuralNetworkParameter)parameters).getHiddenLayer(i) + 1;
            }

            weights.Add(
                new MultiplicationNode(
                    new Tensor(
                        parameters.initializeWeights(
                            currentLength,
                            ((RecurrentNeuralNetworkParameter)parameters).getClassLabelSize(),
                            random),
                        new int[]
                        {
                            currentLength,
                            ((RecurrentNeuralNetworkParameter)parameters).getClassLabelSize()
                        }
                    )
                )
            );

            List<ComputationalNode> currentOldLayers = new List<ComputationalNode>();
            List<ComputationalNode> outputNodes = new List<ComputationalNode>();

            for (int k = 0; k < timeStep; k++)
            {
                this.switches.Add(new Switch());

                List<ComputationalNode> newOldLayers = new List<ComputationalNode>();
                ComputationalNode input = new MultiplicationNode(false, true);
                inputNodes.Add(input);

                ComputationalNode current = input;

                for (int i = 0; i < ((RecurrentNeuralNetworkParameter)parameters).size(); i++)
                {
                    ComputationalNode aw;
                    ComputationalNode aFunction;

                    if (currentOldLayers.Count > 0)
                    {
                        aw = this.addEdge(current, weights[i]);

                        ComputationalNode oWithoutBias = this.addEdge(currentOldLayers[i], new RemoveBias());
                        ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights[i]);
                        ComputationalNode a = this.addAdditionEdge(aw, ou, false);

                        aFunction = this.addEdge(
                            a,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i),
                            true);
                    }
                    else
                    {
                        aw = this.addEdge(current, weights[i], false);

                        aFunction = this.addEdge(
                            aw,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i),
                            true);
                    }

                    current = aFunction;
                    newOldLayers.Add(aFunction);
                }

                currentOldLayers = newOldLayers;

                ComputationalNode node = this.addEdge(current, weights[weights.Count - 1]);
                outputNodes.Add(this.addEdge(node, switches[k]));
            }

            ConcatenatedNode concatenatedNode = (ConcatenatedNode)this.concatEdges(outputNodes, 0);
            this.outputNode = this.addEdge(concatenatedNode, new Softmax());

            ComputationalNode classLabelNode = new ComputationalNode(false, false);
            this.inputNodes.Add(classLabelNode);

            List<ComputationalNode> lossInputs = new List<ComputationalNode>();
            lossInputs.Add(this.outputNode);
            lossInputs.Add(classLabelNode);

            this.addFunctionEdge(lossInputs, parameters.getLossFunction(), false);
            train(trainSet, random);
        }

        protected override List<double> getOutputValue(ComputationalNode outputNode)
        {
            List<double> classLabels = new List<double>();

            for (int i = 0; i < outputNode.getValue().GetShape()[0]; i++)
            {
                int index = -1;
                double max = double.MinValue;

                for (int j = 0; j < outputNode.getValue().GetShape()[1]; j++)
                {
                    if (max < outputNode.getValue().GetValue(new int[] { i, j }))
                    {
                        max = outputNode.getValue().GetValue(new int[] { i, j });
                        index = j;
                    }
                }

                classLabels.Add((double)index);
            }

            return classLabels;
        }

        public override ClassificationPerformance test(List<Tensor> testSet)
        {
            int count = 0;
            int total = 0;

            foreach (Tensor instance in testSet)
            {
                List<int> goldClassLabels = createInputTensors(instance);
                List<double> classLabels = this.predict();

                for (int j = 0; j < (instance.GetShape()[0] / (wordEmbeddingLength + 1)); j++)
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