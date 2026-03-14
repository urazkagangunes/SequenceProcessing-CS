using System;
using System.Collections.Generic;
using ComputationalGraph.Function;
using ComputationalGraph.Initialization;
using ComputationalGraph;

namespace SequenceProcessing.Parameters
{
    [Serializable]
    public class TransformerParameter : NeuralNetworkParameter
    {
        private readonly int L;
        private readonly int N;
        private readonly int V;
        private readonly double epsilon;
        private readonly List<int> inputHiddenLayers;
        private readonly List<int> outputHiddenLayers;
        private readonly List<Function> inputFunctions;
        private readonly List<Function> outputFunctions;
        private readonly List<double> gammaInputValues;
        private readonly List<double> gammaOutputValues;
        private readonly List<double> betaInputValues;
        private readonly List<double> betaOutputValues;

        public TransformerParameter(
            int seed,
            int epoch,
            ComputationalGraph.Optimizer.Optimizer optimizer,
            Initialization initialization,
            Function loss,
            int wordEmbeddingLength,
            int multiHeadAttentionLength,
            int vocabularyLength,
            double epsilon,
            List<int> inputHiddenLayers,
            List<int> outputHiddenLayers,
            List<Function> inputActivationFunctions,
            List<Function> outputActivationFunctions,
            List<double> gammaInputValues,
            List<double> gammaOutputValues,
            List<double> betaInputValues,
            List<double> betaOutputValues)
            : base(seed, epoch, optimizer, initialization, loss, 0.0, 1)
        {
            this.L = wordEmbeddingLength + 1;
            this.N = multiHeadAttentionLength;
            this.V = vocabularyLength;
            this.epsilon = epsilon;
            this.inputHiddenLayers = inputHiddenLayers;
            this.outputHiddenLayers = outputHiddenLayers;
            this.inputFunctions = inputActivationFunctions;
            this.outputFunctions = outputActivationFunctions;
            this.gammaInputValues = gammaInputValues;
            this.gammaOutputValues = gammaOutputValues;
            this.betaInputValues = betaInputValues;
            this.betaOutputValues = betaOutputValues;
        }

        public double getGammaInputValue(int index) => gammaInputValues[index];
        public double getGammaOutputValue(int index) => gammaOutputValues[index];
        public double getBetaInputValue(int index) => betaInputValues[index];
        public double getBetaOutputValue(int index) => betaOutputValues[index];
        public double getEpsilon() => epsilon;
        public int getDk() => L / N;
        public int getL() => L;
        public int getN() => N;
        public int getV() => V;
        public int getInputHiddenLayer(int index) => inputHiddenLayers[index];
        public int getOutputHiddenLayer(int index) => outputHiddenLayers[index];
        public Function getInputActivationFunction(int index) => inputFunctions[index];
        public Function getOutputActivationFunction(int index) => outputFunctions[index];
        public int getInputSize() => inputHiddenLayers.Count;
        public int getOutputSize() => outputHiddenLayers.Count;
    }
}