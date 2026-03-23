using Dictionary.Dictionary;
using Vector = Math.Vector;

namespace SequenceProcessing.Sequence
{
    public class LabelledVectorizedWord : VectorizedWord
    {
        private readonly string _classLabel;

        /**
         * <summary>Creates a labelled vectorized word with the given word, embedding, and class label.</summary>
         *
         * <param name="word">The surface form of the word.</param>
         * <param name="embedding">The embedding vector of the word.</param>
         * <param name="classLabel">The class label of the word.</param>
         */
        public LabelledVectorizedWord(string word, Vector embedding, string classLabel)
            : base(word, embedding)
        {
            _classLabel = classLabel;
        }

        /**
         * <summary>Creates a labelled vectorized word with the given word and class label.</summary>
         *
         * <param name="word">The surface form of the word.</param>
         * <param name="classLabel">The class label of the word.</param>
         */
        public LabelledVectorizedWord(string word, string classLabel)
            : base(word, new Vector(300, 0))
        {
            _classLabel = classLabel;
        }

        /**
         * <summary>Returns the class label of the word.</summary>
         *
         * <returns>The class label of the word.</returns>
         */
        public string GetClassLabel()
        {
            return _classLabel;
        }
    }
}