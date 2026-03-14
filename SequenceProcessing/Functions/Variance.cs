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
        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();
            List<double> variances = new List<double>();

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                double total = 0.0;
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    total += System.Math.Pow(tensor.GetValue(new int[] { i, j }), 2);
                }

                variances.Add(total / tensor.GetShape()[1]);
            }

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    values.Add(variances[i]);
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
                    values.Add(
                        2.0 * System.Math.Sqrt(tensor.GetShape()[1] * tensor.GetValue(new int[] { i, j }))
                        / tensor.GetShape()[1]
                    );
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