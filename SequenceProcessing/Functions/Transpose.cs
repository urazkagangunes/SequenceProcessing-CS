using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Transpose : ComputationalGraph.Function.Function
    {
        /**
         * <summary>Returns the transpose of the given tensor.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <returns>The transposed tensor.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            return tensor.Transpose(new[] { 1, 0 });
        }

        /**
         * <summary>Returns the derivative of the transpose operation for the given backward tensor.</summary>
         *
         * <param name="value">The input tensor.</param>
         * <param name="backward">The backward tensor.</param>
         * <returns>The transposed backward tensor.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            return backward.Transpose(new[] { 1, 0 });
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