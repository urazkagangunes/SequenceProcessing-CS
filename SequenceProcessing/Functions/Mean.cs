using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Mean : ComputationalGraph.Function.Function
    {
        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();
            List<double> means = new List<double>();

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                double total = 0.0;
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    total += tensor.GetValue(new int[] { i, j });
                }

                means.Add(total / tensor.GetShape()[1]);
            }

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    values.Add(means[i]);
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
                    values.Add(1.0 / tensor.GetShape()[1]);
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