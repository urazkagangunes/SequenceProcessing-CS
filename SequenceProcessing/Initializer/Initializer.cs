using Math;

namespace SequenceProcessing.Initializer
{
    public interface Initializer
    {
        /**
         * <summary>Initializes and returns a matrix with the given dimensions.</summary>
         *
         * <param name="row">Number of rows of the matrix.</param>
         * <param name="col">Number of columns of the matrix.</param>
         * <param name="random">Random object used during initialization.</param>
         * <returns>The initialized matrix.</returns>
         */
        Matrix Initialize(int row, int col, System.Random random);
    }
}