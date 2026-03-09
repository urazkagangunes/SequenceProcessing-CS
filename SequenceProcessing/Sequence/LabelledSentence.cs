using Corpus;

namespace SequenceProcessing.Sequence
{
    public class LabelledSentence : Sentence
    {
        private readonly string classLabel;

        public LabelledSentence(string classLabel) : base()
        {
            this.classLabel = classLabel;
        }

        public string GetClassLabel()
        {
            return classLabel;
        }
    }
}