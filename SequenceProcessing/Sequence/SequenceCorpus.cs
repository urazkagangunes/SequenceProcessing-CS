using System.Collections.Generic;
using System.IO;
using Corpus;
using Dictionary.Dictionary;
using Vector = Math.Vector;

namespace SequenceProcessing.Sequence
{
    public class SequenceCorpus : Corpus.Corpus
    {
        /**
         * <summary>
         * Creates a sequence corpus from the given file name by reading the file line by line.
         * It creates a new <see cref="VectorizedWord"/> with the current word and its label.
         * It also creates a new <see cref="Sentence"/> when a new sentence starts and adds each word
         * to that sentence until the sentence ends.
         * </summary>
         *
         * <param name="fileName">File which will be read and parsed.</param>
         */
        public SequenceCorpus(string fileName)
            : base()
        {
            string line;
            Sentence newSentence = null;

            try
            {
                using var br = new StreamReader(File.OpenRead(fileName));
                line = br.ReadLine();

                while (line != null)
                {
                    var items = line.Split(' ');
                    var word = items[0];

                    if (word.Equals("<S>"))
                    {
                        if (items.Length == 2)
                        {
                            newSentence = new LabelledSentence(items[1]);
                        }
                        else
                        {
                            newSentence = new Sentence();
                        }
                    }
                    else
                    {
                        if (word.Equals("</S>"))
                        {
                            AddSentence(newSentence);
                        }
                        else
                        {
                            VectorizedWord newWord;

                            if (items.Length == 2)
                            {
                                newWord = new LabelledVectorizedWord(word, items[1]);
                            }
                            else
                            {
                                newWord = new VectorizedWord(word, new Vector(300, 0));
                            }

                            if (newSentence != null)
                            {
                                newSentence.AddWord(newWord);
                            }
                        }
                    }

                    line = br.ReadLine();
                }
            }
            catch (IOException)
            {
            }
        }

        /**
         * <summary>Returns the distinct class labels in the corpus.</summary>
         *
         * <returns>The list of distinct class labels in the corpus.</returns>
         */
        public List<string> GetClassLabels()
        {
            var sentenceLabelled = false;
            var classLabels = new List<string>();

            if (sentences[0] is LabelledSentence)
            {
                sentenceLabelled = true;
            }

            for (var i = 0; i < SentenceCount(); i++)
            {
                if (sentenceLabelled)
                {
                    var sentence = (LabelledSentence)sentences[i];
                    if (!classLabels.Contains(sentence.GetClassLabel()))
                    {
                        classLabels.Add(sentence.GetClassLabel());
                    }
                }
                else
                {
                    var sentence = sentences[i];
                    for (var j = 0; j < sentence.WordCount(); j++)
                    {
                        var word = (LabelledVectorizedWord)sentence.GetWord(j);
                        if (!classLabels.Contains(word.GetClassLabel()))
                        {
                            classLabels.Add(word.GetClassLabel());
                        }
                    }
                }
            }

            return classLabels;
        }
    }
}