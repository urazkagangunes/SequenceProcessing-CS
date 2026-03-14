using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class MultiplyByConstant : ComputationalGraph.Function.Function
    {
        private readonly double constant;

        public MultiplyByConstant(double constant)
        {
            this.constant = constant;
        }

        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)tensor.GetData();

            foreach (double val in tensorValues)
            {
                double newVal = constant * val;
                values.Add(newVal);
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
                    values.Add(constant);
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