using System.Collections.Generic;
using ComputationalGraph.Function;
using ComputationalGraph.Initialization;
using ComputationalGraph.Optimizer;
using Dictionary.Dictionary;
using NUnit.Framework;
using SequenceProcessing.Classification;
using SequenceProcessing.Parameters;
using Tensor = Math.Tensor;

namespace Test
{
    [TestFixture]
    public class TransformerTest
    {
        /**
         * <summary>Tests whether the transformer model can be initialized and trained without throwing an exception.</summary>
         */
        [Test]
        public void TestInitialization()
        {
            var tensors = new List<Tensor>
            {
                new Tensor(
                    new List<double>
                    {
                        0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41,
                        double.MaxValue,
                        0.27, 0.67, 0.41, 1,
                        0.37, 0.17, 0.41, 6,
                        0.17, 0.65, 0.87, 5,
                        0.97, 0.19, 0.51, 4
                    },
                    new[] { 29 }
                ),
                new Tensor(
                    new List<double>
                    {
                        0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41,
                        double.MaxValue,
                        0.27, 0.67, 0.41, 1,
                        0.37, 0.17, 0.41, 6,
                        0.77, 0.61, 0.27, 2
                    },
                    new[] { 25 }
                ),
                new Tensor(
                    new List<double>
                    {
                        0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9, 0.35, 0.12, 0.27, 0.17, 0.41,
                        double.MaxValue,
                        1.2, 3.6, 7.1, 3,
                        5.4, 0.17, 9.8, 4,
                        0.77, 0.61, 0.27, 2
                    },
                    new[] { 25 }
                )
            };

            var hiddenLayers = new List<int> { 30, 15 };

            var inputFunctions = new List<Function>
            {
                new Tanh(),
                new Sigmoid()
            };

            var outputFunctions = new List<Function>
            {
                new Sigmoid(),
                new Tanh()
            };

            var gammaInputValues = new List<double> { 1.0, 1.0 };
            var gammaOutputValues = new List<double> { 1.0, 1.0, 1.0 };
            var betaInputValues = new List<double> { 0.0, 0.0 };
            var betaOutputValues = new List<double> { 0.0, 0.0, 0.0 };

            var parameter = new TransformerParameter(
                1,
                150,
                new AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
                new RandomInitialization(),
                new CrossEntropyLoss(),
                3,
                2,
                7,
                1e-9,
                hiddenLayers,
                hiddenLayers,
                inputFunctions,
                outputFunctions,
                gammaInputValues,
                gammaOutputValues,
                betaInputValues,
                betaOutputValues);

            var dictionary = new VectorizedDictionary(new DummyWordComparer());
            var transformer = new Transformer(parameter, dictionary);

            Assert.That(() => transformer.Train(tensors), Throws.Nothing);
        }

        /**
         * <summary>Provides a dummy comparer implementation for words used in test initialization.</summary>
         */
        private class DummyWordComparer : IComparer<Word>
        {
            /**
             * <summary>Compares the given words and returns zero for test purposes.</summary>
             *
             * <param name="x">The first word.</param>
             * <param name="y">The second word.</param>
             * <returns>Zero for all comparisons.</returns>
             */
            public int Compare(Word x, Word y)
            {
                return 0;
            }
        }
    }
}