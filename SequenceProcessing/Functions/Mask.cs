using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Mask : ComputationalGraph.Function.Function
    {
        /**
         * <summary>Applies an upper-triangular mask to the given tensor.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <returns>The masked tensor.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            var values = new List<double>();

            for (var i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (var j = 0; j < tensor.GetShape()[1]; j++)
                {
                    if (j > i)
                    {
                        values.Add(double.NegativeInfinity);
                    }
                    else
                    {
                        values.Add(tensor.GetValue(new[] { i, j }));
                    }
                }
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Calculates the derivative of the mask function for the given tensor.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <param name="backward">The backward tensor.</param>
         * <returns>The derivative tensor.</returns>
         */
        public Tensor Derivative(Tensor tensor, Tensor backward)
        {
            var values = new List<double>();

            for (var i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (var j = 0; j < tensor.GetShape()[1]; j++)
                {
                    values.Add(1.0);
                }
            }

            return backward.HadamardProduct(new Tensor(values, tensor.GetShape()));
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