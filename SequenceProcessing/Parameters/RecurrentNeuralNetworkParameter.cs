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
        private readonly List<int> _hiddenLayers;
        private readonly List<Function> _functions;
        private readonly int _classLabelSize;

        /**
         * <summary>Creates a recurrent neural network parameter object.</summary>
         *
         * <param name="seed">Seed value used for random initialization.</param>
         * <param name="epoch">Number of epochs.</param>
         * <param name="optimizer">Optimizer used during training.</param>
         * <param name="initialization">Initialization strategy.</param>
         * <param name="loss">Loss function.</param>
         * <param name="hiddenLayers">Hidden layer sizes.</param>
         * <param name="functions">Activation functions of hidden layers.</param>
         * <param name="classLabelSize">Number of class labels.</param>
         */
        public RecurrentNeuralNetworkParameter(
            int seed,
            int epoch,
            ComputationalGraph.Optimizer.Optimizer optimizer,
            Initialization initialization,
            Function loss,
            List<int> hiddenLayers,
            List<Function> functions,
            int classLabelSize)
            : base(seed, epoch, optimizer, initialization, loss, 0.0, 1)
        {
            _hiddenLayers = hiddenLayers;
            _functions = functions;
            _classLabelSize = classLabelSize;
        }

        /**
         * <summary>Returns the number of hidden layers.</summary>
         *
         * <returns>The number of hidden layers.</returns>
         */
        public int Size()
        {
            return _hiddenLayers.Count;
        }

        /**
         * <summary>Returns the class label size.</summary>
         *
         * <returns>The class label size.</returns>
         */
        public int GetClassLabelSize()
        {
            return _classLabelSize;
        }

        /**
         * <summary>Returns the activation function at the given index.</summary>
         *
         * <param name="index">Index of the activation function.</param>
         * <returns>The activation function at the given index.</returns>
         */
        public Function GetActivationFunction(int index)
        {
            return _functions[index];
        }

        /**
         * <summary>Returns the hidden layer size at the given index.</summary>
         *
         * <param name="index">Index of the hidden layer.</param>
         * <returns>The hidden layer size at the given index.</returns>
         */
        public int GetHiddenLayer(int index)
        {
            return _hiddenLayers[index];
        }
    }
}