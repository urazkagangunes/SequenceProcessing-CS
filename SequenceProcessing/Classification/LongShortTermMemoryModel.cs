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
        public LongShortTermMemoryModel(NeuralNetworkParameter parameter, int wordEmbeddingLength)
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
                for (int j = 0; j < 4; j++)
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
            List<ComputationalNode> currentOldCValues = new List<ComputationalNode>();
            List<ComputationalNode> outputNodes = new List<ComputationalNode>();

            for (int k = 0; k < timeStep; k++)
            {
                this.switches.Add(new Switch());

                List<ComputationalNode> newOldLayers = new List<ComputationalNode>();
                List<ComputationalNode> newOldCValues = new List<ComputationalNode>();

                ComputationalNode input = new MultiplicationNode(false, true);
                inputNodes.Add(input);

                ComputationalNode current = input;

                for (int i = 0; i < weights.Count - 1; i += 4)
                {
                    ComputationalNode aw;
                    ComputationalNode aFunction;
                    ComputationalNode ct;

                    if (currentOldLayers.Count > 0)
                    {
                        aw = this.addEdge(current, weights[i]);

                        ComputationalNode oWithoutBias = this.addEdge(currentOldLayers[i / 4], new RemoveBias());

                        ComputationalNode ou = this.addEdge(oWithoutBias, recurrentWeights[i]);
                        ComputationalNode awOu = this.addAdditionEdge(aw, ou, false);
                        ComputationalNode it = this.addEdge(
                            awOu,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i));

                        aw = this.addEdge(current, weights[i + 1]);
                        ou = this.addEdge(oWithoutBias, recurrentWeights[i + 1]);
                        awOu = this.addAdditionEdge(aw, ou, false);
                        ComputationalNode ft = this.addEdge(
                            awOu,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i + 1));

                        aw = this.addEdge(current, weights[i + 2]);
                        ou = this.addEdge(oWithoutBias, recurrentWeights[i + 2]);
                        awOu = this.addAdditionEdge(aw, ou, false);
                        ComputationalNode ot = this.addEdge(
                            awOu,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i + 2));

                        aw = this.addEdge(current, weights[i + 3]);
                        ou = this.addEdge(oWithoutBias, recurrentWeights[i + 3]);
                        awOu = this.addAdditionEdge(aw, ou, false);
                        ComputationalNode cTemp = this.addEdge(awOu, new Tanh());

                        ComputationalNode ftCt1 = this.addEdge(ft, currentOldCValues[i / 4], false, true);
                        ComputationalNode itCtTemp = this.addEdge(it, cTemp, false, true);
                        ComputationalNode cmb = this.addAdditionEdge(ftCt1, itCtTemp, false);

                        ct = this.addEdge(
                            cmb,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i + 3));

                        ComputationalNode tanhCt = this.addEdge(ct, new Tanh());
                        aFunction = this.addEdge(tanhCt, ot, true, true);
                    }
                    else
                    {
                        aw = this.addEdge(current, weights[i]);
                        ComputationalNode it = this.addEdge(
                            aw,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i));

                        aw = this.addEdge(current, weights[i + 1]);
                        ComputationalNode ot = this.addEdge(
                            aw,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i + 2));

                        aw = this.addEdge(current, weights[i + 3]);
                        ComputationalNode cTemp = this.addEdge(aw, new Tanh());

                        ComputationalNode itCTemp = this.addEdge(it, cTemp, false, true);

                        ct = this.addEdge(
                            itCTemp,
                            ((RecurrentNeuralNetworkParameter)parameters).getActivationFunction(i + 3));

                        ComputationalNode tanhCt = this.addEdge(ct, new Tanh());
                        aFunction = this.addEdge(tanhCt, ot, true, true);
                    }

                    current = aFunction;
                    newOldLayers.Add(aFunction);
                    newOldCValues.Add(ct);
                }

                currentOldLayers = newOldLayers;
                currentOldCValues = newOldCValues;

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