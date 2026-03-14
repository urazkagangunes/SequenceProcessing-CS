using System;
using System.Collections.Generic;
using Classification.Performance;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using DictNS = global::Dictionary;
using SequenceProcessing.Functions;
using SequenceProcessing.Parameters;
using Tensor = Math.Tensor;
using MathVector = global::Math.Vector;

namespace SequenceProcessing.Classification
{
    [Serializable]
    public class Transformer : ComputationalGraph.ComputationalGraph
    {
        private readonly DictNS.Dictionary.VectorizedDictionary dictionary;
        private int startIndex;
        private int endIndex;

        public Transformer(NeuralNetworkParameter parameter, DictNS.Dictionary.VectorizedDictionary dictionary)
            : base(parameter)
        {
            this.dictionary = dictionary;

            for (int k = 0; k < this.dictionary.Size(); k++)
            {
                if (this.dictionary.GetWord(k).GetName().Equals("<S>"))
                {
                    this.startIndex = k;
                }
                else if (this.dictionary.GetWord(k).GetName().Equals("</S>"))
                {
                    this.endIndex = k;
                }
            }
        }

        private Tensor positionalEncoding(Tensor tensor, int wordEmbeddingLength)
        {
            List<double> values = new List<double>();

            for (int i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (int j = 0; j < tensor.GetShape()[1]; j++)
                {
                    double val = tensor.GetValue(new int[] { i, j });

                    if (j % 2 == 0)
                    {
                        values.Add(val + System.Math.Sin((i + 1.0) / System.Math.Pow(10000, (j + 0.0) / wordEmbeddingLength)));
                    }
                    else
                    {
                        values.Add(val + System.Math.Cos((i + 1.0) / System.Math.Pow(10000, (j - 1.0) / wordEmbeddingLength)));
                    }
                }
            }

            return new Tensor(values, tensor.GetShape());
        }

        private List<int> createInputTensors(Tensor instance, ComputationalNode input1, ComputationalNode input2, int wordEmbeddingLength)
        {
            bool isOutput = false;
            int curLength = 0;

            List<int> classLabels = new List<int>();
            List<double> values = new List<double>();

            for (int i = 0; i < instance.GetShape()[0]; i++)
            {
                double val = instance.GetValue(new int[] { i });

                if (val == double.MaxValue)
                {
                    isOutput = true;

                    input1.setValue(new Tensor(values, new int[] { curLength / wordEmbeddingLength, wordEmbeddingLength }));
                    input1.setValue(positionalEncoding(input1.getValue(), wordEmbeddingLength));

                    curLength = 0;
                    values.Clear();
                }
                else if (isOutput)
                {
                    if ((curLength + 1) % (wordEmbeddingLength + 1) == 0)
                    {
                        classLabels.Add((int)val);
                    }
                    else
                    {
                        values.Add(val);
                    }

                    curLength++;
                }
                else
                {
                    values.Add(val);
                    curLength++;
                }
            }

            input2.setValue(new Tensor(values, new int[] { values.Count / wordEmbeddingLength, wordEmbeddingLength }));
            input2.setValue(positionalEncoding(input2.getValue(), wordEmbeddingLength));

            return classLabels;
        }

        private ComputationalNode layerNormalization(ComputationalNode input, TransformerParameter parameter, bool isInput, int[] lnSize)
        {
            List<double> data = new List<double>();

            ComputationalNode inputC1Mean = this.addEdge(input, new Mean());
            ComputationalNode mean1Minus = this.addEdge(inputC1Mean, new Negation());
            ComputationalNode inputC1Mean1Minus = this.addAdditionEdge(input, mean1Minus, false);
            ComputationalNode variance1 = this.addEdge(inputC1Mean1Minus, new Variance());
            ComputationalNode rootVariance1 = this.addEdge(variance1, new SquareRoot(parameter.getEpsilon()));
            ComputationalNode inverseRootVariance1 = this.addEdge(rootVariance1, new Inverse());
            ComputationalNode lnValue1 = this.addEdge(inputC1Mean1Minus, inverseRootVariance1, false, true);

            if (isInput)
            {
                for (int j = 0; j < parameter.getL(); j++)
                {
                    data.Add(parameter.getGammaInputValue(lnSize[0]));
                }
                lnSize[0]++;
            }
            else
            {
                for (int j = 0; j < parameter.getL(); j++)
                {
                    data.Add(parameter.getGammaOutputValue(lnSize[1]));
                }
                lnSize[1]++;
            }

            ComputationalNode gammaInput1 =
                new MultiplicationNode(true, false, new Tensor(data, new int[] { 1, parameter.getL() }), true);

            ComputationalNode lnValue1GammaInput1 = this.addEdge(lnValue1, gammaInput1);

            data.Clear();

            if (isInput)
            {
                for (int j = 0; j < parameter.getL(); j++)
                {
                    data.Add(parameter.getBetaInputValue(lnSize[2]));
                }
                lnSize[2]++;
            }
            else
            {
                for (int j = 0; j < parameter.getL(); j++)
                {
                    data.Add(parameter.getBetaOutputValue(lnSize[3]));
                }
                lnSize[3]++;
            }

            ComputationalNode betaInput1 =
                new ComputationalNode(true, false, new Tensor(data, new int[] { 1, parameter.getL() }));

            return this.addAdditionEdge(lnValue1GammaInput1, betaInput1, false);
        }

        private List<ComputationalNode> multiHeadAttention(ComputationalNode input, TransformerParameter parameter, bool isMasked, Random random)
        {
            List<ComputationalNode> nodes = new List<ComputationalNode>();

            for (int i = 0; i < parameter.getN(); i++)
            {
                ComputationalNode wk =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode k = this.addEdge(input, wk);

                ComputationalNode wq =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode q = this.addEdge(input, wq);

                ComputationalNode wv =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode v = this.addEdge(input, wv);

                ComputationalNode kTranspose = this.addEdge(k, new Transpose());
                ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
                ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / System.Math.Sqrt(parameter.getDk())));

                ComputationalNode sQkDk;
                if (isMasked)
                {
                    ComputationalNode mQkDk = this.addEdge(qkDk, new Mask());
                    sQkDk = this.addEdge(mQkDk, new Softmax());
                }
                else
                {
                    sQkDk = this.addEdge(qkDk, new Softmax());
                }

                ComputationalNode attention = this.addEdge(sQkDk, v);
                nodes.Add(attention);
            }

            return nodes;
        }

        private ComputationalNode feedforwardNeuralNetwork(
            ComputationalNode current,
            int currentLayerSize,
            TransformerParameter parameter,
            Random random,
            bool isInput)
        {
            int size;
            if (isInput)
            {
                size = parameter.getInputSize();
            }
            else
            {
                size = parameter.getOutputSize();
            }

            for (int i = 0; i < size; i++)
            {
                if (isInput)
                {
                    ComputationalNode hiddenWeight =
                        new MultiplicationNode(
                            new Tensor(
                                parameter.initializeWeights(currentLayerSize, parameter.getInputHiddenLayer(i), random),
                                new int[] { currentLayerSize, parameter.getInputHiddenLayer(i) }));

                    ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight);
                    current = this.addEdge(hiddenLayer, parameter.getInputActivationFunction(i), true);
                    currentLayerSize = parameter.getInputHiddenLayer(i) + 1;
                }
                else
                {
                    ComputationalNode hiddenWeight =
                        new MultiplicationNode(
                            new Tensor(
                                parameter.initializeWeights(currentLayerSize, parameter.getOutputHiddenLayer(i), random),
                                new int[] { currentLayerSize, parameter.getOutputHiddenLayer(i) }));

                    ComputationalNode hiddenLayer = this.addEdge(current, hiddenWeight);
                    current = this.addEdge(hiddenLayer, parameter.getOutputActivationFunction(i), true);
                    currentLayerSize = parameter.getOutputHiddenLayer(i) + 1;
                }
            }

            ComputationalNode outputWeight =
                new MultiplicationNode(
                    new Tensor(
                        parameter.initializeWeights(currentLayerSize, parameter.getL(), random),
                        new int[] { currentLayerSize, parameter.getL() }));

            ComputationalNode outputLayer = this.addEdge(current, outputWeight);
            return this.addEdge(outputLayer, new Softmax());
        }

        public override void train(List<Tensor> trainSet)
        {
            TransformerParameter parameter = (TransformerParameter)this.parameters;
            int[] lnSize = new int[4];
            Random random = new Random(parameter.GetSeed());

            // Encoder Block
            ComputationalNode input1 = new MultiplicationNode(false, true);
            this.inputNodes.Add(input1);

            ConcatenatedNode concatenatedNode1 =
                (ConcatenatedNode)this.concatEdges(multiHeadAttention(input1, parameter, false, random), 1);

            ComputationalNode we =
                new MultiplicationNode(
                    new Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getL(), random),
                        new int[] { parameter.getL(), parameter.getL() }));

            ComputationalNode c1 = this.addEdge(concatenatedNode1, we);
            ComputationalNode inputC1 = this.addAdditionEdge(input1, c1, false);
            ComputationalNode y1 = layerNormalization(inputC1, parameter, true, lnSize);
            ComputationalNode oe = this.addAdditionEdge(feedforwardNeuralNetwork(y1, parameter.getL(), parameter, random, true), y1, false);
            ComputationalNode encoder = layerNormalization(oe, parameter, true, lnSize);

            // Decoder Block
            ComputationalNode input2 = new MultiplicationNode(false, true);
            this.inputNodes.Add(input2);

            ConcatenatedNode concatenatedNode2 =
                (ConcatenatedNode)this.concatEdges(multiHeadAttention(input2, parameter, true, random), 1);

            ComputationalNode wd1 =
                new MultiplicationNode(
                    new Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getL(), random),
                        new int[] { parameter.getL(), parameter.getL() }));

            ComputationalNode c2 = this.addEdge(concatenatedNode2, wd1);
            ComputationalNode inputC2 = this.addAdditionEdge(input2, c2, false);
            ComputationalNode cd2 = layerNormalization(inputC2, parameter, false, lnSize);

            List<ComputationalNode> nodes = new List<ComputationalNode>();
            for (int i = 0; i < parameter.getN(); i++)
            {
                ComputationalNode wk =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode k = this.addEdge(encoder, wk);

                ComputationalNode wq =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode q = this.addEdge(cd2, wq);

                ComputationalNode wv =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.initializeWeights(parameter.getL(), parameter.getDk(), random),
                            new int[] { parameter.getL(), parameter.getDk() }));

                ComputationalNode v = this.addEdge(encoder, wv);

                ComputationalNode kTranspose = this.addEdge(k, new Transpose());
                ComputationalNode qk = this.addEdge(q, kTranspose, false, false);
                ComputationalNode qkDk = this.addEdge(qk, new MultiplyByConstant(1.0 / System.Math.Sqrt(parameter.getDk())));
                ComputationalNode sQkDk = this.addEdge(qkDk, new Softmax());
                ComputationalNode attention = this.addEdge(sQkDk, v);

                nodes.Add(attention);
            }

            ConcatenatedNode concatenatedNode3 = (ConcatenatedNode)this.concatEdges(nodes, 1);

            ComputationalNode wd2 =
                new MultiplicationNode(
                    new Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getL(), random),
                        new int[] { parameter.getL(), parameter.getL() }));

            ComputationalNode cd3 = this.addEdge(concatenatedNode3, wd2);
            ComputationalNode cd3cd2 = this.addAdditionEdge(cd2, cd3, false);
            ComputationalNode yd1 = this.layerNormalization(cd3cd2, parameter, false, lnSize);
            ComputationalNode od = this.feedforwardNeuralNetwork(yd1, parameter.getL(), parameter, random, false);
            ComputationalNode oy = this.addAdditionEdge(od, yd1, false);
            ComputationalNode d = this.layerNormalization(oy, parameter, false, lnSize);

            ComputationalNode wdo =
                new MultiplicationNode(
                    new Tensor(
                        parameter.initializeWeights(parameter.getL(), parameter.getV(), random),
                        new int[] { parameter.getL(), parameter.getV() }));

            ComputationalNode decoder = this.addEdge(d, wdo);
            this.outputNode = this.addEdge(decoder, new Softmax());

            ComputationalNode classLabelNode = new ComputationalNode(false, false);
            this.inputNodes.Add(classLabelNode);

            List<ComputationalNode> lossInputs = new List<ComputationalNode>();
            lossInputs.Add(this.outputNode);
            lossInputs.Add(classLabelNode);

            this.addFunctionEdge(lossInputs, parameter.getLossFunction(), false);

            // Training
            for (int i = 0; i < parameter.getEpoch(); i++)
            {
                for (int j = 0; j < trainSet.Count; j++)
                {
                    int i1 = random.Next(trainSet.Count);
                    int i2 = random.Next(trainSet.Count);

                    Tensor tmp = trainSet[i1];
                    trainSet[i1] = trainSet[i2];
                    trainSet[i2] = tmp;
                }

                foreach (Tensor instance in trainSet)
                {
                    List<int> classLabels = createInputTensors(instance, this.inputNodes[0], this.inputNodes[1], parameter.getL() - 1);
                    List<double> classLabelValues = new List<double>();

                    foreach (int classLabel in classLabels)
                    {
                        for (int j = 0; j < parameter.getV(); j++)
                        {
                            if (j == classLabel)
                            {
                                classLabelValues.Add(1.0);
                            }
                            else
                            {
                                classLabelValues.Add(0.0);
                            }
                        }
                    }

                    this.inputNodes[2].setValue(new Tensor(classLabelValues, new int[] { classLabels.Count, parameter.getV() }));
                    this.forwardCalculation();
                    this.backpropagation();
                }

                parameter.getOptimizer().setLearningRate();
            }
        }

        private void setInputNode(int bound, MathVector vector, ComputationalNode node)
        {
            List<double> data = new List<double>();

            if (node.getValue() != null)
            {
                data = new List<double>((List<double>)node.getValue().GetData());
            }

            for (int i = 0; i < vector.Size(); i++)
            {
                if (i % 2 == 0)
                {
                    data.Add(vector.GetValue(i) + System.Math.Sin((bound + 0.0) / System.Math.Pow(10000, (i + 0.0) / vector.Size())));
                }
                else
                {
                    data.Add(vector.GetValue(i) + System.Math.Cos((bound + 0.0) / System.Math.Pow(10000, (i - 1.0) / vector.Size())));
                }
            }

            node.setValue(new Tensor(data, new int[] { bound, vector.Size() }));
        }

        public override ClassificationPerformance test(List<Tensor> testSet)
        {
            int count = 0;
            int total = 0;

            foreach (Tensor instance in testSet)
            {
                List<double> classLabels = null;

                List<int> goldClassLabels =
                    createInputTensors(
                        instance,
                        this.inputNodes[0],
                        new ComputationalNode(false, false),
                        ((DictNS.Dictionary.VectorizedWord)this.dictionary.GetWord(0)).GetVector().Size());

                int j = 1;
                int currentWordIndex = this.startIndex;

                do
                {
                    setInputNode(
                        j,
                        ((DictNS.Dictionary.VectorizedWord)this.dictionary.GetWord(currentWordIndex)).GetVector(),
                        this.inputNodes[1]);

                    classLabels = this.predict();

                    if (goldClassLabels.Count >= classLabels.Count &&
                        (int)classLabels[classLabels.Count - 1] == goldClassLabels[classLabels.Count - 1])
                    {
                        count++;
                    }

                    total++;
                    j++;
                    currentWordIndex = (int)classLabels[classLabels.Count - 1];
                }
                while (currentWordIndex != this.endIndex);

                if (classLabels.Count < goldClassLabels.Count)
                {
                    total += goldClassLabels.Count - classLabels.Count;
                }
            }

            return new ClassificationPerformance((count + 0.0) / total);
        }

        protected override List<double> getOutputValue(ComputationalNode computationalNode)
        {
            List<double> classLabels = new List<double>();
            Tensor value = computationalNode.getValue();

            for (int i = 0; i < value.GetShape()[0]; i++)
            {
                double max = double.MinValue;
                double index = -1;

                for (int j = 0; j < value.GetShape()[1]; j++)
                {
                    if (value.GetValue(new int[] { i, j }) > max)
                    {
                        max = value.GetValue(new int[] { i, j });
                        index = j;
                    }
                }

                classLabels.Add(index);
            }

            return classLabels;
        }
    }
}