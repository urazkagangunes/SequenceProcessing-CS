using System;
using System.Collections.Generic;
using Classification.Parameter;
using Corpus;
using Math;
using SequenceProcessing.Sequence;

namespace SequenceProcessing.Classification
{
    public abstract class Model
    {
        protected SequenceCorpus Corpus;
        protected List<Matrix> Layers;
        protected List<Matrix> OldLayers;
        protected List<Matrix> Weights;
        protected List<Matrix> RecurrentWeights;
        protected List<string> ClassLabels;
        protected ActivationFunction ActivationFunctionType;

        /**
         * <summary>Creates a model with the given corpus, network parameters, and initializer.</summary>
         *
         * <param name="corpus">The sequence corpus used by the model.</param>
         * <param name="parameters">The deep network parameters.</param>
         * <param name="initializer">The initializer used for weight initialization.</param>
         */
        public Model(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer.Initializer initializer)
        {
            Corpus = corpus;
            ActivationFunctionType = parameters.GetActivationFunction();

            var layers = new List<Matrix>();
            var oldLayers = new List<Matrix>();
            var weights = new List<Matrix>();
            var recurrentWeights = new List<Matrix>();

            ClassLabels = corpus.GetClassLabels();

            var inputSize = ((LabelledVectorizedWord)corpus.GetSentence(0).GetWord(0)).GetVector().Size();
            layers.Add(new Matrix(inputSize, 1));

            for (var i = 0; i < parameters.LayerSize(); i++)
            {
                oldLayers.Add(new Matrix(parameters.GetHiddenNodes(i), 1));
                layers.Add(new Matrix(parameters.GetHiddenNodes(i), 1));
                recurrentWeights.Add(
                    initializer.Initialize(
                        parameters.GetHiddenNodes(i),
                        parameters.GetHiddenNodes(i),
                        new Random(parameters.GetSeed())
                    )
                );
            }

            layers.Add(new Matrix(ClassLabels.Count, 1));

            for (var i = 0; i < layers.Count - 1; i++)
            {
                weights.Add(
                    initializer.Initialize(
                        layers[i + 1].GetRow(),
                        layers[i].GetRow() + 1,
                        new Random(parameters.GetSeed())
                    )
                );
            }

            Layers = layers;
            OldLayers = oldLayers;
            Weights = weights;
            RecurrentWeights = recurrentWeights;
        }

        /**
         * <summary>Creates the input vector for the given labelled vectorized word.</summary>
         *
         * <param name="word">The labelled vectorized word.</param>
         */
        protected void CreateInputVector(LabelledVectorizedWord word)
        {
            for (var i = 0; i < Layers[0].GetRow(); i++)
            {
                Layers[0].SetValue(i, 0, word.GetVector().GetValue(i));
            }

            Layers[0] = Biased(Layers[0]);
        }

        /**
         * <summary>Returns the biased version of the given matrix.</summary>
         *
         * <param name="m">The matrix to bias.</param>
         * <returns>The biased matrix.</returns>
         */
        protected Matrix Biased(Matrix m)
        {
            var v = new Matrix(m.GetRow() + 1, m.GetColumn());

            for (var i = 0; i < m.GetRow(); i++)
            {
                v.SetValue(i, 0, m.GetValue(i, 0));
            }

            v.SetValue(m.GetRow(), 0, 1.0);
            return v;
        }

        /**
         * <summary>Updates the old layer values with the current hidden layer values.</summary>
         */
        protected void OldLayersUpdate()
        {
            for (var i = 0; i < OldLayers.Count; i++)
            {
                for (var j = 0; j < OldLayers[i].GetRow(); j++)
                {
                    OldLayers[i].SetValue(j, 0, Layers[i + 1].GetValue(j, 0));
                }
            }
        }

        /**
         * <summary>Sets all layer values to zero.</summary>
         */
        protected void SetLayersValuesToZero()
        {
            for (var j = 0; j < Layers.Count - 1; j++)
            {
                var size = Layers[j].GetRow();
                Layers[j] = new Matrix(size - 1, 1);

                for (var i = 0; i < Layers[j].GetRow(); i++)
                {
                    Layers[j].SetValue(i, 0, 0.0);
                }
            }

            for (var i = 0; i < Layers[Layers.Count - 1].GetRow(); i++)
            {
                Layers[Layers.Count - 1].SetValue(i, 0, 0.0);
            }
        }

        /**
         * <summary>Calculates the one-minus version of the given matrix.</summary>
         *
         * <param name="hidden">The input matrix.</param>
         * <returns>A matrix whose elements are one minus the input matrix values.</returns>
         */
        protected Matrix CalculateOneMinusMatrix(Matrix hidden)
        {
            var oneMinus = new Matrix(hidden.GetRow(), 1);

            for (var i = 0; i < oneMinus.GetRow(); i++)
            {
                oneMinus.SetValue(i, 0, 1.0 - hidden.GetValue(i, 0));
            }

            return oneMinus;
        }

        /**
         * <summary>Normalizes the output layer using softmax.</summary>
         */
        protected void NormalizeOutput()
        {
            var sum = 0.0;
            var values = new double[Layers[Layers.Count - 1].GetRow()];

            for (var i = 0; i < values.Length; i++)
            {
                sum += System.Math.Exp(Layers[Layers.Count - 1].GetValue(i, 0));
            }

            for (var i = 0; i < values.Length; i++)
            {
                values[i] = System.Math.Exp(Layers[Layers.Count - 1].GetValue(i, 0)) / sum;
            }

            for (var i = 0; i < values.Length; i++)
            {
                Layers[Layers.Count - 1].SetValue(i, 0, values[i]);
            }
        }

        /**
         * <summary>Calculates the difference between the reference output and the predicted output.</summary>
         *
         * <param name="word">The labelled vectorized word.</param>
         * <returns>The difference matrix between the reference and predicted outputs.</returns>
         */
        protected Matrix CalculateRMinusY(LabelledVectorizedWord word)
        {
            var r = new Matrix(ClassLabels.Count, 1);
            var index = ClassLabels.IndexOf(word.GetClassLabel());

            r.SetValue(index, 0, 1.0);

            for (var i = 0; i < ClassLabels.Count; i++)
            {
                r.SetValue(i, 0, r.GetValue(i, 0) - Layers[Layers.Count - 1].GetValue(i, 0));
            }

            return r;
        }

        /**
         * <summary>Calculates the derivative of the given matrix according to the activation function.</summary>
         *
         * <param name="matrix">The matrix whose derivative will be calculated.</param>
         * <param name="function">The activation function.</param>
         * <returns>The derivative matrix.</returns>
         */
        protected Matrix Derivative(Matrix matrix, ActivationFunction function)
        {
            if (function.Equals(global::Classification.Parameter.ActivationFunction.SIGMOID))
            {
                var oneMinusHidden = CalculateOneMinusMatrix(matrix);
                return matrix.ElementProduct(oneMinusHidden);
            }

            if (function.Equals(global::Classification.Parameter.ActivationFunction.TANH))
            {
                var oneMinusA2 = new Matrix(matrix.GetRow(), 1);
                var a2 = matrix.ElementProduct(matrix);

                for (var i = 0; i < oneMinusA2.GetRow(); i++)
                {
                    oneMinusA2.SetValue(i, 0, 1.0 - a2.GetValue(i, 0));
                }

                return oneMinusA2;
            }

            var der = new Matrix(matrix.GetRow(), 1);

            for (var i = 0; i < matrix.GetRow(); i++)
            {
                if (matrix.GetValue(i, 0) > 0)
                {
                    der.SetValue(i, 0, 1.0);
                }
            }

            return der;
        }

        /**
         * <summary>Applies the given activation function to the matrix.</summary>
         *
         * <param name="matrix">The input matrix.</param>
         * <param name="function">The activation function to apply.</param>
         * <returns>The activated matrix.</returns>
         */
        protected Matrix ActivationFunction(Matrix matrix, ActivationFunction function)
        {
            var r = new Matrix(matrix.GetRow(), matrix.GetColumn());

            if (function.Equals(global::Classification.Parameter.ActivationFunction.SIGMOID))
            {
                for (var i = 0; i < matrix.GetRow(); i++)
                {
                    r.SetValue(i, 0, 1 / (1 + System.Math.Exp(-matrix.GetValue(i, 0))));
                }
            }
            else if (function.Equals(global::Classification.Parameter.ActivationFunction.TANH))
            {
                for (var i = 0; i < matrix.GetRow(); i++)
                {
                    r.SetValue(i, 0, System.Math.Tanh(matrix.GetValue(i, 0)));
                }
            }
            else
            {
                for (var i = 0; i < matrix.GetRow(); i++)
                {
                    if (matrix.GetValue(i, 0) < 0)
                    {
                        r.SetValue(i, 0, 0.0);
                    }
                    else
                    {
                        r.SetValue(i, 0, matrix.GetValue(i, 0));
                    }
                }
            }

            return r;
        }

        /**
         * <summary>Clears the model state.</summary>
         */
        protected abstract void Clear();

        /**
         * <summary>Clears the old layer values by setting them to zero.</summary>
         */
        protected void ClearOldValues()
        {
            for (var i = 0; i < OldLayers.Count; i++)
            {
                for (var k = 0; k < OldLayers[i].GetRow(); k++)
                {
                    OldLayers[i].SetValue(k, 0, 0.0);
                }
            }
        }

        /**
         * <summary>Calculates the output of the model for the given labelled vectorized word.</summary>
         *
         * <param name="word">The labelled vectorized word.</param>
         */
        protected abstract void CalculateOutput(LabelledVectorizedWord word);

        /**
         * <summary>Predicts the class labels for the given sentence.</summary>
         *
         * <param name="sentence">The sentence to classify.</param>
         * <returns>The predicted class labels.</returns>
         */
        public List<string> Predict(Sentence sentence)
        {
            var classLabels = new List<string>();

            for (var i = 0; i < sentence.WordCount(); i++)
            {
                var word = (LabelledVectorizedWord)sentence.GetWord(i);
                CalculateOutput(word);

                var bestValue = double.MinValue;
                var best = ClassLabels[0];

                for (var j = 0; j < Layers[Layers.Count - 1].GetRow(); j++)
                {
                    if (Layers[Layers.Count - 1].GetValue(j, 0) > bestValue)
                    {
                        bestValue = Layers[Layers.Count - 1].GetValue(j, 0);
                        best = ClassLabels[j];
                    }
                }

                classLabels.Add(best);
                Clear();
            }

            ClearOldValues();
            return classLabels;
        }
    }
}