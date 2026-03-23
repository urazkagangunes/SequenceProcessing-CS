using Corpus;

namespace SequenceProcessing.Sequence
{
    public class LabelledSentence : Sentence
    {
        private readonly string _classLabel;

        /**
         * <summary>Creates a labelled sentence with the given class label.</summary>
         *
         * <param name="classLabel">The class label of the sentence.</param>
         */
        public LabelledSentence(string classLabel)
            : base()
        {
            _classLabel = classLabel;
        }

        /**
         * <summary>Returns the class label of the sentence.</summary>
         *
         * <returns>The class label of the sentence.</returns>
         */
        public string GetClassLabel()
        {
            return _classLabel;
        }
    }
}