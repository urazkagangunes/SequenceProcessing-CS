using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace SequenceProcessing.Functions
{
    [Serializable]
    public class Switch : ComputationalGraph.Function.Function
    {
        private bool turn;

        public Switch()
        {
            this.turn = true;
        }

        public void setTurn(bool turn)
        {
            this.turn = turn;
        }

        public Tensor calculate(Tensor matrix)
        {
            if (this.turn)
            {
                return matrix;
            }

            List<double> values = new List<double>();
            int size = 1;

            for (int i = 0; i < matrix.GetShape().Length; i++)
            {
                size *= matrix.GetShape()[i];
            }

            for (int i = 0; i < size; i++)
            {
                values.Add(0.0);
            }

            return new Tensor(values, matrix.GetShape());
        }

        public Tensor derivative(Tensor value, Tensor backward)
        {
            if (this.turn)
            {
                return backward;
            }

            return calculate(value);
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}