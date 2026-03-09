using Dictionary.Dictionary;
using Vector = Math.Vector;

namespace SequenceProcessing.Sequence
{
    public class LabelledVectorizedWord : VectorizedWord
    {
        private readonly string classLabel;

        public LabelledVectorizedWord(string word, Vector embedding, string classLabel)
            : base(word, embedding)
        {
            this.classLabel = classLabel;
        }

        public LabelledVectorizedWord(string word, string classLabel)
            : base(word, new Vector(300, 0))
        {
            this.classLabel = classLabel;
        }

        public string GetClassLabel()
        {
            return classLabel;
        }
    }
}