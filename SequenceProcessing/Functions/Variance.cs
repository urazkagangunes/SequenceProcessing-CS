using System;
using System.Collections.Generic;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Variance : Function
    {
        /**
         * <summary>Calculates the row-wise variance-like values of the given tensor and returns them in tensor form.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <returns>The tensor whose rows are filled with the corresponding variance values.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            var values = new List<double>();
            var variances = new List<double>();

            for (var i = 0; i < tensor.GetShape()[0]; i++)
            {
                var total = 0.0;
                for (var j = 0; j < tensor.GetShape()[1]; j++)
                {
                    total += System.Math.Pow(tensor.GetValue(new[] { i, j }), 2);
                }

                variances.Add(total / tensor.GetShape()[1]);
            }

            for (var i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (var j = 0; j < tensor.GetShape()[1]; j++)
                {
                    values.Add(variances[i]);
                }
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Calculates the derivative of the variance function for the given tensor.</summary>
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
                    values.Add(
                        2.0 * System.Math.Sqrt(tensor.GetShape()[1] * tensor.GetValue(new[] { i, j }))
                        / tensor.GetShape()[1]
                    );
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