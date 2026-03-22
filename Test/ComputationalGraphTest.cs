using System;
using System.Collections.Generic;
using NUnit.Framework;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Initialization;
using ComputationalGraph.Node;
using ComputationalGraph.Optimizer;
using Tensor = Math.Tensor;
using CGInitialization = ComputationalGraph.Initialization.Initialization;

namespace Test
{
    public class ComputationalGraphTest
    {
        private static Tensor Tensor1D(params double[] values)
        {
            return new Tensor(new List<double>(values), new[] { values.Length });
        }

        private static Tensor Tensor2D(int rows, int cols, params double[] values)
        {
            Assert.AreEqual(rows * cols, values.Length);
            return new Tensor(new List<double>(values), new[] { rows, cols });
        }

        private static List<double> Data(Tensor tensor)
        {
            return (List<double>)tensor.GetData();
        }

        private static void AssertTensor(Tensor actual, double tolerance, params double[] expected)
        {
            List<double> actualValues = Data(actual);
            Assert.AreEqual(expected.Length, actualValues.Count);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.That(actualValues[i], Is.EqualTo(expected[i]).Within(tolerance), $"Mismatch at index {i}");
            }
        }

        private sealed class InspectableSgd : StochasticGradientDescent
        {
            public InspectableSgd(double learningRate, double etaDecrease)
                : base(learningRate, etaDecrease)
            {
            }

            public void ApplyGradients(ComputationalNode node)
            {
                setGradients(node);
            }

            public double LearningRateValue => learningRate;
        }

        private sealed class InspectableMomentum : SGDMomentum
        {
            public InspectableMomentum(double learningRate, double etaDecrease, double momentum)
                : base(learningRate, etaDecrease, momentum)
            {
            }

            public void ApplyGradients(ComputationalNode node)
            {
                setGradients(node);
            }

            public double LearningRateValue => learningRate;

            public double[] VelocityOf(ComputationalNode node)
            {
                return velocityMap[node];
            }
        }

        private sealed class FixedInitialization : CGInitialization
        {
            public List<double> Initialize(int row, int column, Random random)
            {
                List<double> result = new List<double>();
                for (int i = 0; i < row * column; i++)
                {
                    result.Add(10 + i);
                }

                return result;
            }
        }

        [Test]
        public void TestComputationalNodeAdd()
        {
            var parent = new ComputationalNode();
            var child = new ComputationalNode();

            parent.add(child);

            Assert.AreEqual(1, parent.childrenSize());
            Assert.AreEqual(1, child.parentsSize());
            Assert.AreSame(child, parent.getChild(0));
            Assert.AreSame(parent, child.getParent(0));
        }

        [Test]
        public void TestConcatenatedNodeIndexing()
        {
            var c = new ConcatenatedNode(1);
            var n1 = new ComputationalNode();
            var n2 = new ComputationalNode();
            var n3 = new ComputationalNode();

            c.addNode(n1);
            c.addNode(n2);
            c.addNode(n3);

            Assert.AreEqual(1, c.getDimension());
            Assert.AreEqual(0, c.getIndex(n1));
            Assert.AreEqual(1, c.getIndex(n2));
            Assert.AreEqual(2, c.getIndex(n3));
        }

        [Test]
        public void TestFunctionNodeStoresFunction()
        {
            var function = new ReLU();
            var node = new FunctionNode(true, function);

            Assert.AreSame(function, node.getFunction());
            Assert.IsTrue(node.isBiasedNode());
        }

        [Test]
        public void TestMultiplicationNodeStoresHadamardAndPriority()
        {
            var priority = new ComputationalNode();
            var node = new MultiplicationNode(false, true, true, priority);

            Assert.IsTrue(node.isHadamard());
            Assert.AreSame(priority, node.getPriorityNode());
            Assert.IsTrue(node.isBiasedNode());
        }

        [Test]
        public void TestReLU()
        {
            var relu = new ReLU();

            var output = relu.calculate(Tensor1D(-2.0, 0.0, 3.0));
            AssertTensor(output, 1e-10, 0.0, 0.0, 3.0);

            var grad = relu.derivative(output, Tensor1D(1.0, 2.0, 3.0));
            AssertTensor(grad, 1e-10, 0.0, 0.0, 3.0);
        }

        [Test]
        public void TestSigmoid()
        {
            var sigmoid = new Sigmoid();

            var output = sigmoid.calculate(Tensor1D(0.0));
            AssertTensor(output, 1e-10, 0.5);

            var grad = sigmoid.derivative(output, Tensor1D(2.0));
            AssertTensor(grad, 1e-10, 0.5);
        }

        [Test]
        public void TestTanh()
        {
            var tanh = new Tanh();

            var output = tanh.calculate(Tensor1D(0.0, 1.0));
            double t1 = System.Math.Tanh(1.0);

            AssertTensor(output, 1e-10, 0.0, t1);

            var grad = tanh.derivative(output, Tensor1D(2.0, 3.0));
            AssertTensor(grad, 1e-10, 2.0, (1 - t1 * t1) * 3.0);
        }

        [Test]
        public void TestNegation()
        {
            var negation = new Negation();

            var output = negation.calculate(Tensor1D(2.0, -3.0));
            AssertTensor(output, 1e-10, -2.0, 3.0);

            var grad = negation.derivative(output, Tensor1D(4.0, -5.0));
            AssertTensor(grad, 1e-10, -4.0, 5.0);
        }

        [Test]
        public void TestLogarithm()
        {
            var log = new Logarithm();

            var output = log.calculate(Tensor1D(1.0, System.Math.E));
            AssertTensor(output, 1e-10, 0.0, 1.0);

            var grad = log.derivative(output, Tensor1D(2.0, 3.0));
            AssertTensor(grad, 1e-10, 2.0, 3.0 / System.Math.E);
        }

        [Test]
        public void TestPower()
        {
            var power = new Power();

            var output = power.calculate(Tensor1D(2.0, -3.0));
            AssertTensor(output, 1e-10, 4.0, 9.0);

            var grad = power.derivative(output, Tensor1D(1.0, 2.0));
            AssertTensor(grad, 1e-10, 4.0, 12.0);
        }

        [Test]
        public void TestElu()
        {
            var elu = new ELU();

            var output = elu.calculate(Tensor1D(-1.0, 2.0));
            double neg = System.Math.Exp(-1.0) - 1.0;

            AssertTensor(output, 1e-10, neg, 2.0);

            var grad = elu.derivative(output, Tensor1D(3.0, 4.0));
            AssertTensor(grad, 1e-10, (neg + 1.0) * 3.0, 4.0);
        }

        [Test]
        public void TestDelu()
        {
            var delu = new DELU();

            var output = delu.calculate(Tensor1D(0.0, 2.0));
            AssertTensor(output, 1e-10, 0.0, 2.0);

            var grad = delu.derivative(output, Tensor1D(4.0, 5.0));
            AssertTensor(grad, 1e-10, 2.0, 5.0);
        }

        [Test]
        public void TestDropoutAtZeroProbabilityActsAsIdentity()
        {
            var dropout = new Dropout(0.0, new Random(1));
            var input = Tensor1D(1.0, 2.0, 3.0);

            var output = dropout.calculate(input);
            AssertTensor(output, 1e-10, 1.0, 2.0, 3.0);

            var grad = dropout.derivative(output, Tensor1D(4.0, 5.0, 6.0));
            AssertTensor(grad, 1e-10, 4.0, 5.0, 6.0);
        }

        [Test]
        public void TestSoftmax()
        {
            var softmax = new Softmax();
            var input = Tensor2D(2, 2, 0.0, 0.0, 1.0, 2.0);

            var output = softmax.calculate(input);

            double e1 = System.Math.Exp(1.0);
            double e2 = System.Math.Exp(2.0);
            double sum = e1 + e2;

            AssertTensor(output, 1e-10, 0.5, 0.5, e1 / sum, e2 / sum);

            List<double> outData = Data(output);
            Assert.That(outData[0] + outData[1], Is.EqualTo(1.0).Within(1e-10));
            Assert.That(outData[2] + outData[3], Is.EqualTo(1.0).Within(1e-10));
        }

        [Test]
        public void TestSoftmaxDerivative()
        {
            var softmax = new Softmax();
            var probabilities = Tensor1D(0.25, 0.75);
            var backward = Tensor1D(1.0, 0.0);

            var grad = softmax.derivative(probabilities, backward);

            AssertTensor(grad, 1e-10, 0.1875, -0.1875);
        }

        [Test]
        public void TestReLUAddEdge()
        {
            var input = new ComputationalNode();
            var relu = new ReLU();

            ComputationalNode output = relu.addEdge(new List<ComputationalNode> { input }, true);

            Assert.That(output, Is.TypeOf<FunctionNode>());
            Assert.AreEqual(1, input.childrenSize());
            Assert.AreSame(output, input.getChild(0));
            Assert.AreEqual(1, output.parentsSize());
            Assert.AreSame(input, output.getParent(0));
            Assert.IsTrue(output.isBiasedNode());
        }

        [Test]
        public void TestCrossEntropyLossAddEdge()
        {
            var predicted = new ComputationalNode();
            var gold = new ComputationalNode();
            var loss = new CrossEntropyLoss();

            ComputationalNode output = loss.addEdge(new List<ComputationalNode> { predicted, gold }, true);

            Assert.That(output, Is.TypeOf<MultiplicationNode>());
            Assert.IsTrue(((MultiplicationNode)output).isHadamard());
            Assert.IsTrue(output.isBiasedNode());

            Assert.AreEqual(1, predicted.childrenSize());
            Assert.That(predicted.getChild(0), Is.TypeOf<FunctionNode>());

            Assert.AreEqual(1, gold.childrenSize());
            Assert.AreSame(output, gold.getChild(0));

            Assert.AreEqual(2, output.parentsSize());
        }

        [Test]
        public void TestMeanSquaredErrorLossAddEdge()
        {
            var first = new ComputationalNode();
            var second = new ComputationalNode();
            var loss = new MeanSquaredErrorLoss();

            ComputationalNode output = loss.addEdge(new List<ComputationalNode> { first, second }, false);

            Assert.That(output, Is.TypeOf<FunctionNode>());
            Assert.AreEqual(1, first.childrenSize());
            Assert.That(first.getChild(0), Is.TypeOf<FunctionNode>());

            ComputationalNode negated = first.getChild(0);
            Assert.AreEqual(1, negated.childrenSize());

            ComputationalNode diffNode = negated.getChild(0);
            Assert.AreSame(diffNode, second.getChild(0));

            Assert.AreEqual(1, diffNode.childrenSize());
            Assert.AreSame(output, diffNode.getChild(0));
        }

        [Test]
        public void TestSiLUAddEdge()
        {
            var input = new ComputationalNode();
            var silu = new SiLU();

            ComputationalNode output = silu.addEdge(new List<ComputationalNode> { input }, true);

            Assert.That(output, Is.TypeOf<MultiplicationNode>());
            Assert.IsTrue(((MultiplicationNode)output).isHadamard());
            Assert.IsTrue(output.isBiasedNode());

            Assert.AreEqual(2, input.childrenSize());
            Assert.That(input.getChild(0), Is.TypeOf<FunctionNode>());
            Assert.AreSame(output, input.getChild(1));
            Assert.AreEqual(2, output.parentsSize());
        }

        [Test]
        public void TestTanhShrinkAddEdge()
        {
            var input = new ComputationalNode();
            var tanhShrink = new TanhShrink();

            ComputationalNode output = tanhShrink.addEdge(new List<ComputationalNode> { input }, true);

            Assert.That(output, Is.TypeOf<ComputationalNode>());
            Assert.IsTrue(output.isBiasedNode());

            Assert.AreEqual(2, input.childrenSize());
            Assert.That(input.getChild(0), Is.TypeOf<FunctionNode>());
            Assert.AreSame(output, input.getChild(1));

            ComputationalNode tanhNode = input.getChild(0);
            Assert.AreEqual(1, tanhNode.childrenSize());

            ComputationalNode negativeTanh = tanhNode.getChild(0);
            Assert.That(negativeTanh, Is.TypeOf<FunctionNode>());
            Assert.AreEqual(1, negativeTanh.childrenSize());
            Assert.AreSame(output, negativeTanh.getChild(0));
            Assert.AreEqual(2, output.parentsSize());
        }

        [Test]
        public void TestRandomInitialization()
        {
            var initialization = new RandomInitialization();
            var values = initialization.Initialize(2, 3, new Random(7));

            Assert.AreEqual(6, values.Count);

            foreach (double value in values)
            {
                Assert.That(value, Is.InRange(-0.01, 0.01));
            }
        }

        [Test]
        public void TestUniformXavierInitialization()
        {
            int row = 2;
            int column = 4;
            double limit = System.Math.Sqrt(6.0 / (row + column));

            var initialization = new UniformXavierInitialization();
            var values = initialization.Initialize(row, column, new Random(11));

            Assert.AreEqual(8, values.Count);

            foreach (double value in values)
            {
                Assert.That(value, Is.InRange(-limit, limit));
            }
        }

        [Test]
        public void TestHeUniformInitialization()
        {
            int row = 3;
            int column = 5;
            double min = -System.Math.Sqrt(6.0 / row);
            double max = System.Math.Sqrt(6.0 / column);

            var initialization = new HeUniformInitialization();
            var values = initialization.Initialize(row, column, new Random(13));

            Assert.AreEqual(15, values.Count);

            foreach (double value in values)
            {
                Assert.That(value, Is.GreaterThanOrEqualTo(min).And.LessThanOrEqualTo(max));
            }
        }

        [Test]
        public void TestInitializationIsRepeatableWithSameSeed()
        {
            var init1 = new UniformXavierInitialization();
            var init2 = new UniformXavierInitialization();

            var values1 = init1.Initialize(2, 2, new Random(42));
            var values2 = init2.Initialize(2, 2, new Random(42));

            Assert.AreEqual(values1.Count, values2.Count);
            for (int i = 0; i < values1.Count; i++)
            {
                Assert.That(values1[i], Is.EqualTo(values2[i]).Within(1e-12));
            }
        }

        [Test]
        public void TestStochasticGradientDescentScalesBackward()
        {
            var optimizer = new InspectableSgd(0.1, 0.9);
            var node = new ComputationalNode(true, false, Tensor1D(1.0, 2.0));
            node.setBackward(Tensor1D(5.0, -3.0));

            optimizer.ApplyGradients(node);

            AssertTensor(node.getBackward(), 1e-10, 0.5, -0.3);
        }

        [Test]
        public void TestOptimizerLearningRateDecay()
        {
            var optimizer = new InspectableSgd(0.5, 0.1);

            optimizer.setLearningRate();

            Assert.That(optimizer.LearningRateValue, Is.EqualTo(0.05).Within(1e-12));
        }

        [Test]
        public void TestSgdMomentumFirstStep()
        {
            var optimizer = new InspectableMomentum(0.1, 0.9, 0.9);
            var node = new ComputationalNode(true, false, Tensor1D(0.0, 0.0));
            node.setBackward(Tensor1D(1.0, -2.0));

            optimizer.ApplyGradients(node);

            AssertTensor(node.getBackward(), 1e-10, 0.01, -0.02);

            double[] velocity = optimizer.VelocityOf(node);
            Assert.That(velocity[0], Is.EqualTo(0.1).Within(1e-12));
            Assert.That(velocity[1], Is.EqualTo(-0.2).Within(1e-12));
        }

        [Test]
        public void TestSgdMomentumSecondStepUsesPreviousVelocity()
        {
            var optimizer = new InspectableMomentum(0.1, 0.9, 0.9);
            var node = new ComputationalNode(true, false, Tensor1D(0.0, 0.0));

            node.setBackward(Tensor1D(1.0, -2.0));
            optimizer.ApplyGradients(node);

            node.setBackward(Tensor1D(3.0, 1.0));
            optimizer.ApplyGradients(node);

            AssertTensor(node.getBackward(), 1e-10, 0.039, -0.008);

            double[] velocity = optimizer.VelocityOf(node);
            Assert.That(velocity[0], Is.EqualTo(0.39).Within(1e-12));
            Assert.That(velocity[1], Is.EqualTo(-0.08).Within(1e-12));
        }

        [Test]
        public void TestNeuralNetworkParameterFullConstructor()
        {
            var optimizer = new StochasticGradientDescent(0.01, 0.95);
            var initialization = new FixedInitialization();
            var lossFunction = new MeanSquaredErrorLoss();

            var parameter = new NeuralNetworkParameter(
                123,
                20,
                optimizer,
                initialization,
                lossFunction,
                0.3,
                16);

            Assert.AreSame(optimizer, parameter.getOptimizer());
            Assert.AreEqual(20, parameter.getEpoch());
            Assert.That(parameter.getDropout(), Is.EqualTo(0.3).Within(1e-12));
            Assert.AreSame(lossFunction, parameter.getLossFunction());
            Assert.AreEqual(16, parameter.getBatchSize());

            CollectionAssert.AreEqual(
                new List<double> { 10, 11, 12, 13 },
                parameter.initializeWeights(2, 2, new Random(1)));
        }

        [Test]
        public void TestNeuralNetworkParameterDefaultValues()
        {
            var optimizer = new StochasticGradientDescent(0.01, 0.95);
            var parameter = new NeuralNetworkParameter(5, 7, optimizer);

            Assert.AreSame(optimizer, parameter.getOptimizer());
            Assert.AreEqual(7, parameter.getEpoch());
            Assert.That(parameter.getDropout(), Is.EqualTo(0.0).Within(1e-12));
            Assert.That(parameter.getLossFunction(), Is.TypeOf<CrossEntropyLoss>());
            Assert.AreEqual(1, parameter.getBatchSize());

            List<double> weights = parameter.initializeWeights(2, 2, new Random(1));
            Assert.AreEqual(4, weights.Count);

            foreach (double weight in weights)
            {
                Assert.That(weight, Is.InRange(-0.01, 0.01));
            }
        }

        [Test]
        public void TestNeuralNetworkParameterAlternativeConstructor()
        {
            var optimizer = new StochasticGradientDescent(0.02, 0.9);
            var loss = new MeanSquaredErrorLoss();

            var parameter = new NeuralNetworkParameter(9, 3, optimizer, loss, 0.25);

            Assert.AreSame(optimizer, parameter.getOptimizer());
            Assert.AreEqual(3, parameter.getEpoch());
            Assert.AreSame(loss, parameter.getLossFunction());
            Assert.That(parameter.getDropout(), Is.EqualTo(0.25).Within(1e-12));
            Assert.AreEqual(1, parameter.getBatchSize());
        }
    }
}