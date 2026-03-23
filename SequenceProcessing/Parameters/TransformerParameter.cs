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
        private readonly int _l;
        private readonly int _n;
        private readonly int _v;
        private readonly double _epsilon;
        private readonly List<int> _inputHiddenLayers;
        private readonly List<int> _outputHiddenLayers;
        private readonly List<Function> _inputFunctions;
        private readonly List<Function> _outputFunctions;
        private readonly List<double> _gammaInputValues;
        private readonly List<double> _gammaOutputValues;
        private readonly List<double> _betaInputValues;
        private readonly List<double> _betaOutputValues;

        /**
         * <summary>Creates a transformer parameter object.</summary>
         *
         * <param name="seed">Seed value used for random initialization.</param>
         * <param name="epoch">Number of epochs.</param>
         * <param name="optimizer">Optimizer used during training.</param>
         * <param name="initialization">Initialization strategy.</param>
         * <param name="loss">Loss function.</param>
         * <param name="wordEmbeddingLength">Word embedding length.</param>
         * <param name="multiHeadAttentionLength">Number of attention heads.</param>
         * <param name="vocabularyLength">Vocabulary size.</param>
         * <param name="epsilon">Epsilon value used in normalization.</param>
         * <param name="inputHiddenLayers">Input-side hidden layer sizes.</param>
         * <param name="outputHiddenLayers">Output-side hidden layer sizes.</param>
         * <param name="inputActivationFunctions">Input-side activation functions.</param>
         * <param name="outputActivationFunctions">Output-side activation functions.</param>
         * <param name="gammaInputValues">Gamma values for input-side layer normalization.</param>
         * <param name="gammaOutputValues">Gamma values for output-side layer normalization.</param>
         * <param name="betaInputValues">Beta values for input-side layer normalization.</param>
         * <param name="betaOutputValues">Beta values for output-side layer normalization.</param>
         */
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
            _l = wordEmbeddingLength + 1;
            _n = multiHeadAttentionLength;
            _v = vocabularyLength;
            _epsilon = epsilon;
            _inputHiddenLayers = inputHiddenLayers;
            _outputHiddenLayers = outputHiddenLayers;
            _inputFunctions = inputActivationFunctions;
            _outputFunctions = outputActivationFunctions;
            _gammaInputValues = gammaInputValues;
            _gammaOutputValues = gammaOutputValues;
            _betaInputValues = betaInputValues;
            _betaOutputValues = betaOutputValues;
        }

        /**
         * <summary>Returns the gamma input value at the given index.</summary>
         *
         * <param name="index">Index of the gamma input value.</param>
         * <returns>The gamma input value at the given index.</returns>
         */
        public double GetGammaInputValue(int index)
        {
            return _gammaInputValues[index];
        }

        /**
         * <summary>Returns the gamma output value at the given index.</summary>
         *
         * <param name="index">Index of the gamma output value.</param>
         * <returns>The gamma output value at the given index.</returns>
         */
        public double GetGammaOutputValue(int index)
        {
            return _gammaOutputValues[index];
        }

        /**
         * <summary>Returns the beta input value at the given index.</summary>
         *
         * <param name="index">Index of the beta input value.</param>
         * <returns>The beta input value at the given index.</returns>
         */
        public double GetBetaInputValue(int index)
        {
            return _betaInputValues[index];
        }

        /**
         * <summary>Returns the beta output value at the given index.</summary>
         *
         * <param name="index">Index of the beta output value.</param>
         * <returns>The beta output value at the given index.</returns>
         */
        public double GetBetaOutputValue(int index)
        {
            return _betaOutputValues[index];
        }

        /**
         * <summary>Returns the epsilon value used in normalization.</summary>
         *
         * <returns>The epsilon value.</returns>
         */
        public double GetEpsilon()
        {
            return _epsilon;
        }

        /**
         * <summary>Returns the attention head dimension.</summary>
         *
         * <returns>The attention head dimension.</returns>
         */
        public int GetDk()
        {
            return _l / _n;
        }

        /**
         * <summary>Returns the model dimension.</summary>
         *
         * <returns>The model dimension.</returns>
         */
        public int GetL()
        {
            return _l;
        }

        /**
         * <summary>Returns the number of attention heads.</summary>
         *
         * <returns>The number of attention heads.</returns>
         */
        public int GetN()
        {
            return _n;
        }

        /**
         * <summary>Returns the vocabulary size.</summary>
         *
         * <returns>The vocabulary size.</returns>
         */
        public int GetV()
        {
            return _v;
        }

        /**
         * <summary>Returns the input hidden layer size at the given index.</summary>
         *
         * <param name="index">Index of the input hidden layer.</param>
         * <returns>The input hidden layer size at the given index.</returns>
         */
        public int GetInputHiddenLayer(int index)
        {
            return _inputHiddenLayers[index];
        }

        /**
         * <summary>Returns the output hidden layer size at the given index.</summary>
         *
         * <param name="index">Index of the output hidden layer.</param>
         * <returns>The output hidden layer size at the given index.</returns>
         */
        public int GetOutputHiddenLayer(int index)
        {
            return _outputHiddenLayers[index];
        }

        /**
         * <summary>Returns the input activation function at the given index.</summary>
         *
         * <param name="index">Index of the input activation function.</param>
         * <returns>The input activation function at the given index.</returns>
         */
        public Function GetInputActivationFunction(int index)
        {
            return _inputFunctions[index];
        }

        /**
         * <summary>Returns the output activation function at the given index.</summary>
         *
         * <param name="index">Index of the output activation function.</param>
         * <returns>The output activation function at the given index.</returns>
         */
        public Function GetOutputActivationFunction(int index)
        {
            return _outputFunctions[index];
        }

        /**
         * <summary>Returns the number of input hidden layers.</summary>
         *
         * <returns>The number of input hidden layers.</returns>
         */
        public int GetInputSize()
        {
            return _inputHiddenLayers.Count;
        }

        /**
         * <summary>Returns the number of output hidden layers.</summary>
         *
         * <returns>The number of output hidden layers.</returns>
         */
        public int GetOutputSize()
        {
            return _outputHiddenLayers.Count;
        }
    }
}