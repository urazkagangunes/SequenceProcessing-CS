using Math;

namespace SequenceProcessing.Initializer
{
    public class Random : Initializer
    {
        /**
         * <summary>Initializes and returns a matrix with random values in the given range.</summary>
         *
         * <param name="row">Number of rows of the matrix.</param>
         * <param name="col">Number of columns of the matrix.</param>
         * <param name="random">Random object used during initialization.</param>
         * <returns>The initialized matrix.</returns>
         */
        public Matrix Initialize(int row, int col, System.Random random)
        {
            return new Matrix(row, col, -0.01, +0.01, random);
        }
    }
}