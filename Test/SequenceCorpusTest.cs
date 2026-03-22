using System.IO;
using NUnit.Framework;
using SequenceProcessing.Sequence;

namespace Test
{
    public class SequenceCorpusTest
    {
        private static string DataPath(string fileName)
        {
            return Path.GetFullPath(
                Path.Combine(TestContext.CurrentContext.TestDirectory, "..", "..", "..", "Resources", fileName));
        }

        [Test]
        public void TestCorpus01()
        {
            var corpus = new SequenceCorpus(DataPath("disambiguation-penn.txt"));
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus02()
        {
            var corpus = new SequenceCorpus(DataPath("postag-atis-en.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(61879, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus03()
        {
            var corpus = new SequenceCorpus(DataPath("slot-atis-en.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(61879, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus04()
        {
            var corpus = new SequenceCorpus(DataPath("slot-atis-tr.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus05()
        {
            var corpus = new SequenceCorpus(DataPath("disambiguation-atis.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus06()
        {
            var corpus = new SequenceCorpus(DataPath("metamorpheme-atis.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus07()
        {
            var corpus = new SequenceCorpus(DataPath("postag-atis-tr.txt"));
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus08()
        {
            var corpus = new SequenceCorpus(DataPath("metamorpheme-penn.txt"));
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus09()
        {
            var corpus = new SequenceCorpus(DataPath("ner-penn.txt"));
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus10()
        {
            var corpus = new SequenceCorpus(DataPath("postag-penn.txt"));
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus11()
        {
            var corpus = new SequenceCorpus(DataPath("semanticrolelabeling-penn.txt"));
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus12()
        {
            var corpus = new SequenceCorpus(DataPath("semantics-penn.txt"));
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus13()
        {
            var corpus = new SequenceCorpus(DataPath("shallowparse-penn.txt"));
            Assert.AreEqual(9557, corpus.SentenceCount());
            Assert.AreEqual(87279, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus14()
        {
            var corpus = new SequenceCorpus(DataPath("disambiguation-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus15()
        {
            var corpus = new SequenceCorpus(DataPath("metamorpheme-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus16()
        {
            var corpus = new SequenceCorpus(DataPath("postag-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus17()
        {
            var corpus = new SequenceCorpus(DataPath("semantics-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus18()
        {
            var corpus = new SequenceCorpus(DataPath("shallowparse-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus19()
        {
            var corpus = new SequenceCorpus(DataPath("disambiguation-kenet.txt"));
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus20()
        {
            var corpus = new SequenceCorpus(DataPath("metamorpheme-kenet.txt"));
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus21()
        {
            var corpus = new SequenceCorpus(DataPath("postag-kenet.txt"));
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus22()
        {
            var corpus = new SequenceCorpus(DataPath("disambiguation-framenet.txt"));
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus23()
        {
            var corpus = new SequenceCorpus(DataPath("metamorpheme-framenet.txt"));
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus24()
        {
            var corpus = new SequenceCorpus(DataPath("postag-framenet.txt"));
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus25()
        {
            var corpus = new SequenceCorpus(DataPath("semanticrolelabeling-framenet.txt"));
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus26()
        {
            var corpus = new SequenceCorpus(DataPath("sentiment-tourism.txt"));
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }
    }
}