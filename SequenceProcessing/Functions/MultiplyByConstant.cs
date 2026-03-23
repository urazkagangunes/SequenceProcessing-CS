using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class MultiplyByConstant : ComputationalGraph.Function.Function
    {
        private readonly double _constant;

        /**
         * <summary>Creates a multiply-by-constant function with the given constant value.</summary>
         *
         * <param name="constant">The constant value used in multiplication.</param>
         */
        public MultiplyByConstant(double constant)
        {
            _constant = constant;
        }

        /**
         * <summary>Calculates the output tensor by multiplying each element with the constant value.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <returns>The resulting tensor after multiplication.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            var values = new List<double>();
            var tensorValues = (List<double>)tensor.GetData();

            foreach (var val in tensorValues)
            {
                var newVal = _constant * val;
                values.Add(newVal);
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Calculates the derivative of the multiply-by-constant function.</summary>
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
                    values.Add(_constant);
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