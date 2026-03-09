using System;
using System.Collections.Generic;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Initialization;

namespace SequenceProcessing.Parameters
{
    [Serializable]
    public class RecurrentNeuralNetworkParameter : NeuralNetworkParameter
    {
        private readonly List<int> hiddenLayers;
        private readonly List<Function> functions;
        private readonly int classLabelSize;

        public RecurrentNeuralNetworkParameter(
            int seed,
            int epoch,
            ComputationalGraph.Optimizer.Optimizer optimizer,
            Initialization initialization,
            Function loss,
            List<int> hiddenLayers,
            List<Function> functions,
            int classLabelSize)
            : base(seed, epoch, optimizer, initialization)
        {
            this.hiddenLayers = hiddenLayers;
            this.functions = functions;
            this.classLabelSize = classLabelSize;
        }

        public int size()
        {
            return hiddenLayers.Count;
        }

        public int getClassLabelSize()
        {
            return classLabelSize;
        }

        public Function getActivationFunction(int index)
        {
            return functions[index];
        }

        public int getHiddenLayer(int index)
        {
            return hiddenLayers[index];
        }
    }
}