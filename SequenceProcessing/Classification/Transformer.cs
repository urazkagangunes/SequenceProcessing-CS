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
        private readonly DictNS.Dictionary.VectorizedDictionary _dictionary;
        private int _startIndex;
        private int _endIndex;

        /**
         * <summary>Creates a transformer model with the given parameter set and dictionary.</summary>
         *
         * <param name="parameter">The neural network parameters.</param>
         * <param name="dictionary">The vectorized dictionary.</param>
         */
        public Transformer(NeuralNetworkParameter parameter, DictNS.Dictionary.VectorizedDictionary dictionary)
            : base(parameter)
        {
            _dictionary = dictionary;

            for (var k = 0; k < _dictionary.Size(); k++)
            {
                if (_dictionary.GetWord(k).GetName().Equals("<S>"))
                {
                    _startIndex = k;
                }
                else if (_dictionary.GetWord(k).GetName().Equals("</S>"))
                {
                    _endIndex = k;
                }
            }
        }

        /**
         * <summary>Applies positional encoding to the given tensor.</summary>
         *
         * <param name="tensor">The input tensor.</param>
         * <param name="wordEmbeddingLength">The word embedding length.</param>
         * <returns>The positionally encoded tensor.</returns>
         */
        private Tensor PositionalEncoding(Tensor tensor, int wordEmbeddingLength)
        {
            var values = new List<double>();

            for (var i = 0; i < tensor.GetShape()[0]; i++)
            {
                for (var j = 0; j < tensor.GetShape()[1]; j++)
                {
                    var value = tensor.GetValue(new[] { i, j });

                    if (j % 2 == 0)
                    {
                        values.Add(value + System.Math.Sin((i + 1.0) / System.Math.Pow(10000, (j + 0.0) / wordEmbeddingLength)));
                    }
                    else
                    {
                        values.Add(value + System.Math.Cos((i + 1.0) / System.Math.Pow(10000, (j - 1.0) / wordEmbeddingLength)));
                    }
                }
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Creates input tensors for the encoder and decoder, and returns the class labels.</summary>
         *
         * <param name="instance">The input instance.</param>
         * <param name="input1">The encoder input node.</param>
         * <param name="input2">The decoder input node.</param>
         * <param name="wordEmbeddingLength">The word embedding length.</param>
         * <returns>The class labels extracted from the input.</returns>
         */
        private List<int> CreateInputTensors(
            Tensor instance,
            ComputationalNode input1,
            ComputationalNode input2,
            int wordEmbeddingLength)
        {
            var isOutput = false;
            var currentLength = 0;

            var classLabels = new List<int>();
            var values = new List<double>();

            for (var i = 0; i < instance.GetShape()[0]; i++)
            {
                var value = instance.GetValue(new[] { i });

                if (value == double.MaxValue)
                {
                    isOutput = true;

                    input1.SetValue(new Tensor(values, new[] { currentLength / wordEmbeddingLength, wordEmbeddingLength }));
                    input1.SetValue(PositionalEncoding(input1.GetValue(), wordEmbeddingLength));

                    currentLength = 0;
                    values.Clear();
                }
                else if (isOutput)
                {
                    if ((currentLength + 1) % (wordEmbeddingLength + 1) == 0)
                    {
                        classLabels.Add((int)value);
                    }
                    else
                    {
                        values.Add(value);
                    }

                    currentLength++;
                }
                else
                {
                    values.Add(value);
                    currentLength++;
                }
            }

            input2.SetValue(new Tensor(values, new[] { values.Count / wordEmbeddingLength, wordEmbeddingLength }));
            input2.SetValue(PositionalEncoding(input2.GetValue(), wordEmbeddingLength));

            return classLabels;
        }

        /**
         * <summary>Applies layer normalization to the given computational node.</summary>
         *
         * <param name="input">The input node.</param>
         * <param name="parameter">The transformer parameter set.</param>
         * <param name="isInput">Indicates whether the normalization is for the encoder side.</param>
         * <param name="layerNormalizationSize">The normalization index counters.</param>
         * <returns>The normalized node.</returns>
         */
        private ComputationalNode LayerNormalization(
            ComputationalNode input,
            TransformerParameter parameter,
            bool isInput,
            int[] layerNormalizationSize)
        {
            var data = new List<double>();

            var inputMean = this.AddEdge(input, new Mean());
            var negativeMean = this.AddEdge(inputMean, new Negation());
            var centeredInput = this.AddAdditionEdge(input, negativeMean, false);
            var variance = this.AddEdge(centeredInput, new Variance());
            var rootVariance = this.AddEdge(variance, new SquareRoot(parameter.GetEpsilon()));
            var inverseRootVariance = this.AddEdge(rootVariance, new Inverse());
            var normalizedValue = this.AddEdge(centeredInput, inverseRootVariance, false, true);

            if (isInput)
            {
                for (var j = 0; j < parameter.GetL(); j++)
                {
                    data.Add(parameter.GetGammaInputValue(layerNormalizationSize[0]));
                }

                layerNormalizationSize[0]++;
            }
            else
            {
                for (var j = 0; j < parameter.GetL(); j++)
                {
                    data.Add(parameter.GetGammaOutputValue(layerNormalizationSize[1]));
                }

                layerNormalizationSize[1]++;
            }

            var gammaNode =
                new MultiplicationNode(true, false, new Tensor(data, new[] { 1, parameter.GetL() }), true);

            var normalizedGamma = this.AddEdge(normalizedValue, gammaNode);

            data.Clear();

            if (isInput)
            {
                for (var j = 0; j < parameter.GetL(); j++)
                {
                    data.Add(parameter.GetBetaInputValue(layerNormalizationSize[2]));
                }

                layerNormalizationSize[2]++;
            }
            else
            {
                for (var j = 0; j < parameter.GetL(); j++)
                {
                    data.Add(parameter.GetBetaOutputValue(layerNormalizationSize[3]));
                }

                layerNormalizationSize[3]++;
            }

            var betaNode =
                new ComputationalNode(true, false, new Tensor(data, new[] { 1, parameter.GetL() }));

            return this.AddAdditionEdge(normalizedGamma, betaNode, false);
        }

        /**
         * <summary>Creates multi-head attention outputs for the given input.</summary>
         *
         * <param name="input">The input node.</param>
         * <param name="parameter">The transformer parameter set.</param>
         * <param name="isMasked">Indicates whether masked attention is used.</param>
         * <param name="random">The random generator.</param>
         * <returns>The attention nodes.</returns>
         */
        private List<ComputationalNode> MultiHeadAttention(
            ComputationalNode input,
            TransformerParameter parameter,
            bool isMasked,
            Random random)
        {
            var nodes = new List<ComputationalNode>();

            for (var i = 0; i < parameter.GetN(); i++)
            {
                var wk =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var k = this.AddEdge(input, wk);

                var wq =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var q = this.AddEdge(input, wq);

                var wv =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var v = this.AddEdge(input, wv);

                var kTranspose = this.AddEdge(k, new Transpose());
                var qk = this.AddEdge(q, kTranspose, false, false);
                var qkDk = this.AddEdge(qk, new MultiplyByConstant(1.0 / System.Math.Sqrt(parameter.GetDk())));

                ComputationalNode softmaxNode;
                if (isMasked)
                {
                    var maskedNode = this.AddEdge(qkDk, new Mask());
                    softmaxNode = this.AddEdge(maskedNode, new Softmax());
                }
                else
                {
                    softmaxNode = this.AddEdge(qkDk, new Softmax());
                }

                var attention = this.AddEdge(softmaxNode, v);
                nodes.Add(attention);
            }

            return nodes;
        }

        /**
         * <summary>Creates the feedforward neural network block.</summary>
         *
         * <param name="current">The current node.</param>
         * <param name="currentLayerSize">The current layer size.</param>
         * <param name="parameter">The transformer parameter set.</param>
         * <param name="random">The random generator.</param>
         * <param name="isInput">Indicates whether the input-side network is used.</param>
         * <returns>The output node of the feedforward block.</returns>
         */
        private ComputationalNode FeedforwardNeuralNetwork(
            ComputationalNode current,
            int currentLayerSize,
            TransformerParameter parameter,
            Random random,
            bool isInput)
        {
            var size = isInput ? parameter.GetInputSize() : parameter.GetOutputSize();

            for (var i = 0; i < size; i++)
            {
                if (isInput)
                {
                    var hiddenWeight =
                        new MultiplicationNode(
                            new Tensor(
                                parameter.InitializeWeights(currentLayerSize, parameter.GetInputHiddenLayer(i), random),
                                new[] { currentLayerSize, parameter.GetInputHiddenLayer(i) }));

                    var hiddenLayer = this.AddEdge(current, hiddenWeight);
                    current = this.AddEdge(hiddenLayer, parameter.GetInputActivationFunction(i), true);
                    currentLayerSize = parameter.GetInputHiddenLayer(i) + 1;
                }
                else
                {
                    var hiddenWeight =
                        new MultiplicationNode(
                            new Tensor(
                                parameter.InitializeWeights(currentLayerSize, parameter.GetOutputHiddenLayer(i), random),
                                new[] { currentLayerSize, parameter.GetOutputHiddenLayer(i) }));

                    var hiddenLayer = this.AddEdge(current, hiddenWeight);
                    current = this.AddEdge(hiddenLayer, parameter.GetOutputActivationFunction(i), true);
                    currentLayerSize = parameter.GetOutputHiddenLayer(i) + 1;
                }
            }

            var outputWeight =
                new MultiplicationNode(
                    new Tensor(
                        parameter.InitializeWeights(currentLayerSize, parameter.GetL(), random),
                        new[] { currentLayerSize, parameter.GetL() }));

            var outputLayer = this.AddEdge(current, outputWeight);
            return this.AddEdge(outputLayer, new Softmax());
        }

        /**
         * <summary>Trains the transformer model with the given training set.</summary>
         *
         * <param name="trainSet">The training set.</param>
         */
        public override void Train(List<Tensor> trainSet)
        {
            var parameter = (TransformerParameter)this.Parameters;
            var layerNormalizationSize = new int[4];
            var random = new Random(parameter.GetSeed());

            var input1 = new MultiplicationNode(false, true);
            this.InputNodes.Add(input1);

            var concatenatedNode1 =
                (ConcatenatedNode)this.ConcatEdges(MultiHeadAttention(input1, parameter, false, random), 1);

            var encoderWeight =
                new MultiplicationNode(
                    new Tensor(
                        parameter.InitializeWeights(parameter.GetL(), parameter.GetL(), random),
                        new[] { parameter.GetL(), parameter.GetL() }));

            var c1 = this.AddEdge(concatenatedNode1, encoderWeight);
            var inputC1 = this.AddAdditionEdge(input1, c1, false);
            var y1 = LayerNormalization(inputC1, parameter, true, layerNormalizationSize);
            var oe = this.AddAdditionEdge(
                FeedforwardNeuralNetwork(y1, parameter.GetL(), parameter, random, true),
                y1,
                false);
            var encoder = LayerNormalization(oe, parameter, true, layerNormalizationSize);

            var input2 = new MultiplicationNode(false, true);
            this.InputNodes.Add(input2);

            var concatenatedNode2 =
                (ConcatenatedNode)this.ConcatEdges(MultiHeadAttention(input2, parameter, true, random), 1);

            var decoderWeight1 =
                new MultiplicationNode(
                    new Tensor(
                        parameter.InitializeWeights(parameter.GetL(), parameter.GetL(), random),
                        new[] { parameter.GetL(), parameter.GetL() }));

            var c2 = this.AddEdge(concatenatedNode2, decoderWeight1);
            var inputC2 = this.AddAdditionEdge(input2, c2, false);
            var cd2 = LayerNormalization(inputC2, parameter, false, layerNormalizationSize);

            var nodes = new List<ComputationalNode>();
            for (var i = 0; i < parameter.GetN(); i++)
            {
                var wk =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var k = this.AddEdge(encoder, wk);

                var wq =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var q = this.AddEdge(cd2, wq);

                var wv =
                    new MultiplicationNode(
                        new Tensor(
                            parameter.InitializeWeights(parameter.GetL(), parameter.GetDk(), random),
                            new[] { parameter.GetL(), parameter.GetDk() }));

                var v = this.AddEdge(encoder, wv);

                var kTranspose = this.AddEdge(k, new Transpose());
                var qk = this.AddEdge(q, kTranspose, false, false);
                var qkDk = this.AddEdge(qk, new MultiplyByConstant(1.0 / System.Math.Sqrt(parameter.GetDk())));
                var softmaxNode = this.AddEdge(qkDk, new Softmax());
                var attention = this.AddEdge(softmaxNode, v);

                nodes.Add(attention);
            }

            var concatenatedNode3 = (ConcatenatedNode)this.ConcatEdges(nodes, 1);

            var decoderWeight2 =
                new MultiplicationNode(
                    new Tensor(
                        parameter.InitializeWeights(parameter.GetL(), parameter.GetL(), random),
                        new[] { parameter.GetL(), parameter.GetL() }));

            var cd3 = this.AddEdge(concatenatedNode3, decoderWeight2);
            var cd3Cd2 = this.AddAdditionEdge(cd2, cd3, false);
            var yd1 = this.LayerNormalization(cd3Cd2, parameter, false, layerNormalizationSize);
            var od = this.FeedforwardNeuralNetwork(yd1, parameter.GetL(), parameter, random, false);
            var oy = this.AddAdditionEdge(od, yd1, false);
            var decoderBlock = this.LayerNormalization(oy, parameter, false, layerNormalizationSize);

            var decoderOutputWeight =
                new MultiplicationNode(
                    new Tensor(
                        parameter.InitializeWeights(parameter.GetL(), parameter.GetV(), random),
                        new[] { parameter.GetL(), parameter.GetV() }));

            var decoder = this.AddEdge(decoderBlock, decoderOutputWeight);
            this.OutputNode = this.AddEdge(decoder, new Softmax());

            var classLabelNode = new ComputationalNode(false, false);
            this.InputNodes.Add(classLabelNode);

            var lossInputs = new List<ComputationalNode>();
            lossInputs.Add(this.OutputNode);
            lossInputs.Add(classLabelNode);

            this.AddFunctionEdge(lossInputs, parameter.GetLossFunction(), false);

            for (var i = 0; i < parameter.GetEpoch(); i++)
            {
                for (var j = 0; j < trainSet.Count; j++)
                {
                    var i1 = random.Next(trainSet.Count);
                    var i2 = random.Next(trainSet.Count);

                    var temp = trainSet[i1];
                    trainSet[i1] = trainSet[i2];
                    trainSet[i2] = temp;
                }

                foreach (var instance in trainSet)
                {
                    var classLabels = CreateInputTensors(
                        instance,
                        this.InputNodes[0],
                        this.InputNodes[1],
                        parameter.GetL() - 1);

                    var classLabelValues = new List<double>();

                    foreach (var classLabel in classLabels)
                    {
                        for (var j = 0; j < parameter.GetV(); j++)
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

                    this.InputNodes[2].SetValue(new Tensor(classLabelValues, new[] { classLabels.Count, parameter.GetV() }));
                    this.ForwardCalculation();
                    this.Backpropagation();
                }

                parameter.GetOptimizer().SetLearningRate();
            }
        }

        /**
         * <summary>Sets the input node value using the given vector and positional encoding.</summary>
         *
         * <param name="bound">The current bound value.</param>
         * <param name="vector">The input vector.</param>
         * <param name="node">The computational node.</param>
         */
        private void SetInputNode(int bound, MathVector vector, ComputationalNode node)
        {
            var data = new List<double>();

            if (node.GetValue() != null)
            {
                data = new List<double>((List<double>)node.GetValue().GetData());
            }

            for (var i = 0; i < vector.Size(); i++)
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

            node.SetValue(new Tensor(data, new[] { bound, vector.Size() }));
        }

        /**
         * <summary>Tests the transformer model with the given test set.</summary>
         *
         * <param name="testSet">The test set.</param>
         * <returns>The classification performance.</returns>
         */
        public override ClassificationPerformance Test(List<Tensor> testSet)
        {
            var count = 0;
            var total = 0;

            foreach (var instance in testSet)
            {
                List<double> classLabels = null;

                var goldClassLabels =
                    CreateInputTensors(
                        instance,
                        this.InputNodes[0],
                        new ComputationalNode(false, false),
                        ((DictNS.Dictionary.VectorizedWord)_dictionary.GetWord(0)).GetVector().Size());

                var j = 1;
                var currentWordIndex = _startIndex;

                do
                {
                    SetInputNode(
                        j,
                        ((DictNS.Dictionary.VectorizedWord)_dictionary.GetWord(currentWordIndex)).GetVector(),
                        this.InputNodes[1]);

                    classLabels = this.Predict();

                    if (goldClassLabels.Count >= classLabels.Count &&
                        (int)classLabels[classLabels.Count - 1] == goldClassLabels[classLabels.Count - 1])
                    {
                        count++;
                    }

                    total++;
                    j++;
                    currentWordIndex = (int)classLabels[classLabels.Count - 1];
                }
                while (currentWordIndex != _endIndex);

                if (classLabels.Count < goldClassLabels.Count)
                {
                    total += goldClassLabels.Count - classLabels.Count;
                }
            }

            return new ClassificationPerformance((count + 0.0) / total);
        }

        /**
         * <summary>Returns the predicted class labels from the given computational node.</summary>
         *
         * <param name="computationalNode">The output computational node.</param>
         * <returns>The predicted class labels.</returns>
         */
        protected override List<double> GetOutputValue(ComputationalNode computationalNode)
        {
            var classLabels = new List<double>();
            var value = computationalNode.GetValue();

            for (var i = 0; i < value.GetShape()[0]; i++)
            {
                var max = double.MinValue;
                var index = -1.0;

                for (var j = 0; j < value.GetShape()[1]; j++)
                {
                    if (value.GetValue(new[] { i, j }) > max)
                    {
                        max = value.GetValue(new[] { i, j });
                        index = j;
                    }
                }

                classLabels.Add(index);
            }

            return classLabels;
        }
    }
}