using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Switch : ComputationalGraph.Function.Function
    {
        private bool _turn;

        /**
         * <summary>Creates a switch function and initializes it as active.</summary>
         */
        public Switch()
        {
            _turn = true;
        }

        /**
         * <summary>Sets the state of the switch.</summary>
         *
         * <param name="turn">The new state of the switch.</param>
         */
        public void SetTurn(bool turn)
        {
            _turn = turn;
        }

        /**
         * <summary>Returns the input tensor if the switch is active; otherwise returns a zero tensor of the same shape.</summary>
         *
         * <param name="matrix">The input tensor.</param>
         * <returns>The original tensor or a zero tensor depending on the switch state.</returns>
         */
        public Tensor Calculate(Tensor matrix)
        {
            if (_turn)
            {
                return matrix;
            }

            var values = new List<double>();
            var size = 1;

            for (var i = 0; i < matrix.GetShape().Length; i++)
            {
                size *= matrix.GetShape()[i];
            }

            for (var i = 0; i < size; i++)
            {
                values.Add(0.0);
            }

            return new Tensor(values, matrix.GetShape());
        }

        /**
         * <summary>Returns the backward tensor if the switch is active; otherwise returns a zero tensor of the same shape.</summary>
         *
         * <param name="value">The input tensor.</param>
         * <param name="backward">The backward tensor.</param>
         * <returns>The backward tensor or a zero tensor depending on the switch state.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            if (_turn)
            {
                return backward;
            }

            return Calculate(value);
        }

        /**
         * <summary>Adds a new function node to the graph and returns the created node.</summary>
         *
         * <param name="inputNodes">The input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The newly created computational node.</returns>
         */
        public ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var newNode = new FunctionNode(isBiased, this);
            inputNodes[0].Add(newNode);
            return newNode;
        }
    }
}