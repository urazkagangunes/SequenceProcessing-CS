using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class RemoveBias : ComputationalGraph.Function.Function
    {
        /**
         * <summary>Removes the bias value from the given tensor.</summary>
         *
         * <param name="matrix">The input tensor containing the bias value.</param>
         * <returns>The tensor without the bias value.</returns>
         */
        public Tensor Calculate(Tensor matrix)
        {
            var data = (List<double>)matrix.GetData();
            var values = new List<double>();

            for (var i = 0; i < data.Count - 1; i++)
            {
                values.Add(data[i]);
            }

            return new Tensor(values, new[] { 1, values.Count });
        }

        /**
         * <summary>Calculates the derivative of the remove-bias function.</summary>
         *
         * <param name="value">The input tensor.</param>
         * <param name="backward">The backward tensor.</param>
         * <returns>The derivative tensor with the bias gradient appended.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            var values = (List<double>)backward.GetData();
            var newValues = new List<double>(values);
            newValues.Add(0.0);

            return new Tensor(newValues, new[] { 1, newValues.Count });
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