using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class AdditionByConstant : ComputationalGraph.Function.Function
    {
        private readonly double constant;

        public AdditionByConstant(double constant)
        {
            this.constant = constant;
        }

        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)tensor.GetData();

            foreach (double val in tensorValues)
            {
                values.Add(val + constant);
            }

            return new Tensor(values, tensor.GetShape());
        }

        public Tensor derivative(Tensor tensor, Tensor backward)
        {
            return backward;
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}