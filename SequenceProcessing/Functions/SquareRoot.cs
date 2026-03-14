using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class SquareRoot : ComputationalGraph.Function.Function
    {
        private readonly double epsilon;

        public SquareRoot(double epsilon)
        {
            this.epsilon = epsilon;
        }

        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    values.Add(System.Math.Sqrt(this.epsilon + tensor.GetValue(new int[] { i, j })));
                }
            }

            return new Tensor(values, tensor.GetShape());
        }

        public Tensor derivative(Tensor tensor, Tensor backward)
        {
            List<double> values = new List<double>();

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    double val = tensor.GetValue(new int[] { i, j });
                    values.Add(1.0 / (2.0 * val));
                }
            }

            return backward.HadamardProduct(new Tensor(values, tensor.GetShape()));
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}