using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class RemoveBias : ComputationalGraph.Function.Function
    {
        public Tensor calculate(Tensor matrix)
        {
            List<double> data = (List<double>)matrix.GetData();
            List<double> values = new List<double>();

            for (int i = 0; i < data.Count - 1; i++)
            {
                values.Add(data[i]);
            }

            return new Tensor(values, new int[] { 1, values.Count });
        }

        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = (List<double>)backward.GetData();
            List<double> newValues = new List<double>(values);
            newValues.Add(0.0);

            return new Tensor(newValues, new int[] { 1, newValues.Count });
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}