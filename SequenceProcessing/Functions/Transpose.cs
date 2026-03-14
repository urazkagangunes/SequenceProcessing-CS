using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Transpose : ComputationalGraph.Function.Function
    {
        public Tensor calculate(Tensor tensor)
        {
            return tensor.Transpose(new int[] { 1, 0 });
        }

        public Tensor derivative(Tensor value, Tensor backward)
        {
            return backward.Transpose(new int[] { 1, 0 });
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}