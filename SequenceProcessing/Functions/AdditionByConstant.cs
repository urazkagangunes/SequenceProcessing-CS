using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class AdditionByConstant : ComputationalGraph.Function.Function
    {
        private readonly double _constant;

        /**
         * <summary>Creates an addition-by-constant function with the given constant value.</summary>
         *
         * <param name="constant">The constant value to be added to each tensor element.</param>
         */
        public AdditionByConstant(double constant)
        {
            _constant = constant;
        }

        /**
         * <summary>Calculates the output tensor by adding the constant value to each element.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <returns>The resulting tensor after addition.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            var values = new List<double>();
            var tensorValues = (List<double>)tensor.GetData();

            foreach (var val in tensorValues)
            {
                values.Add(val + _constant);
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Returns the derivative of the function for the given backward tensor.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <param name="backward">The backward tensor.</param>
         * <returns>The derivative tensor.</returns>
         */
        public Tensor Derivative(Tensor tensor, Tensor backward)
        {
            return backward;
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