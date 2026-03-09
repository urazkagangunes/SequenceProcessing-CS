using System.Collections.Generic;
using System.IO;
using Corpus;
using Dictionary.Dictionary;
using Vector = Math.Vector;

namespace SequenceProcessing.Sequence
{
    public class SequenceCorpus : Corpus.Corpus
    {
        /// <summary>
        /// Constructor which takes a file name <see cref="string"/> as an input and reads the file line by line.
        /// It takes each word of the line, and creates a new <see cref="VectorizedWord"/> with current word and its label.
        /// It also creates a new <see cref="Sentence"/> when a new sentence starts, and adds each word to this sentence
        /// till the end of that sentence.
        /// </summary>
        /// <param name="fileName">File which will be read and parsed.</param>
        public SequenceCorpus(string fileName) : base()
        {
            string line, word;
            VectorizedWord newWord;
            Sentence newSentence = null;

            try
            {
                using StreamReader br = new StreamReader(File.OpenRead(fileName));
                line = br.ReadLine();

                while (line != null)
                {
                    string[] items = line.Split(' ');
                    word = items[0];

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

        public List<string> GetClassLabels()
        {
            bool sentenceLabelled = false;
            List<string> classLabels = new List<string>();

            if (sentences[0] is LabelledSentence)
            {
                sentenceLabelled = true;
            }

            for (int i = 0; i < SentenceCount(); i++)
            {
                if (sentenceLabelled)
                {
                    LabelledSentence sentence = (LabelledSentence)sentences[i];
                    if (!classLabels.Contains(sentence.GetClassLabel()))
                    {
                        classLabels.Add(sentence.GetClassLabel());
                    }
                }
                else
                {
                    Sentence sentence = sentences[i];
                    for (int j = 0; j < sentence.WordCount(); j++)
                    {
                        LabelledVectorizedWord word = (LabelledVectorizedWord)sentence.GetWord(j);
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