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
        public GatedRecurrentUnitModel(ComputationalGraph.NeuralNetworkParameter parameter, int wordEmbeddingLength)
            : base(parameter, wordEmbeddingLength)
        {
            this.switches = new List<Switch>();
        }

        public override void train(List<Tensor> trainSet)
        {
            Random random = new Random(parameters.GetSeed());
            int timeStep = findTimeStep(trainSet);

            List<ComputationalNode> weights = new List<ComputationalNode>();
            List<ComputationalNode> recurrentWeights = new List<ComputationalNode>();

            int currentLength = wordEmbeddingLength + 1;

            for (int i = 0; i < ((RecurrentNeuralNetworkParameter)parameters).size(); i++)
            {
                for (int j = 0; j < 3; j++)
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
                }

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
                        aw = this.addEdge(current, weights[i * 3]);

                        ComputationalNode oWithoutBias = this.addEdge(currentOldLayers[i], new RemoveBias());
                        ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights[i * 3]);
                        ComputationalNode awOu = this.addAdditionEdge(aw, ou, false);

                        ComputationalNode zt = this.addEdge(
                            awOu,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i * 2));

                        aw = this.addEdge(current, weights[(i * 3) + 1]);
                        ou = this.addEdge(oWithoutBias, recurrentWeights[(i * 3) + 1]);
                        awOu = this.addAdditionEdge(aw, ou, false);

                        ComputationalNode rt = this.addEdge(
                            awOu,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction((i * 2) + 1));

                        aw = this.addEdge(current, weights[(i * 3) + 2]);
                        ComputationalNode rtHt1 = this.addEdge(rt, oWithoutBias, false, true);
                        ou = this.addEdge(rtHt1, recurrentWeights[(i * 3) + 2]);
                        awOu = this.addAdditionEdge(aw, ou, false);

                        ComputationalNode hTemp = this.addEdge(awOu, new Tanh());
                        ComputationalNode minusZt = this.addEdge(zt, new Negation());
                        ComputationalNode oneMinusZt = this.addEdge(minusZt, new AdditionByConstant(1.0));

                        aw = this.addEdge(oneMinusZt, oWithoutBias, false, true);
                        ou = this.addEdge(hTemp, zt, false, true);
                        aFunction = this.addAdditionEdge(aw, ou, true);
                    }
                    else
                    {
                        aw = this.addEdge(current, weights[i * 3]);
                        ComputationalNode zt = this.addEdge(
                            aw,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i * 2));

                        aw = this.addEdge(current, weights[(i * 3) + 2]);
                        ComputationalNode hTemp = this.addEdge(aw, new Tanh());

                        aFunction = this.addEdge(zt, hTemp, true, true);
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
    }
}