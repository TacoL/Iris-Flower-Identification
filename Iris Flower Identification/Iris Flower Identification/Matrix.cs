using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Iris_Flower_Identification
{
    public class Matrix
    {
        public double[,] matrix;
        public int numRows;
        public int numColumns;

        public static Random r = new Random();
        public Matrix(int numRows, int numColumns)
        {
            matrix = new double[numRows, numColumns];
            this.numRows = numRows;
            this.numColumns = numColumns;
        }

        //1D matrix
        public Matrix(double[] vector)
        {
            matrix = new double[vector.Length, 1];
            for (int i = 0; i < vector.Length; i++)
            {
                matrix[i, 0] = vector[i];
            }
            numRows = vector.Length;
            numColumns = 1;
        }

        public Matrix(double[,] matrix)
        {
            this.matrix = matrix;
            numRows = matrix.GetLength(0);
            numColumns = matrix.GetLength(1);
        }

        public static Matrix Randomize(Matrix a)
        {
            for (int row = 0; row < a.matrix.GetLength(0); row++)
            {
                for (int column = 0; column < a.matrix.GetLength(1); column++)
                {
                    a.matrix[row, column] = r.NextDouble() * (r.Next(0, 100) > 50 ? -1 : 1);
                }
            }

            return a;
        }

        public double[] GetColumn(int columnIdx)
        {
            return Enumerable.Range(0, matrix.GetLength(0)).Select(x => matrix[x, columnIdx]).ToArray();
        }

        public double[] GetRow(int rowIdx)
        {
            return Enumerable.Range(0, matrix.GetLength(1)).Select(x => matrix[rowIdx, x]).ToArray();
        }

        public static Matrix Addition(Matrix a, Matrix b)
        {
            if ((a.numRows == b.numRows) && (a.numColumns == b.numColumns))
            {
                Matrix newMatrix = new Matrix(a.numRows, a.numColumns);
                for (int row = 0; row < a.numRows; row++)
                {
                    for (int column = 0; column < a.numColumns; column++)
                    {
                        newMatrix.matrix[row, column] = a.matrix[row, column] + b.matrix[row, column];
                    }
                }

                return newMatrix;
            }

            return new Matrix(0, 0);
        }

        public static Matrix Subtract(Matrix a, Matrix b)
        {
            if ((a.numRows == b.numRows) && (a.numColumns == b.numColumns))
            {
                Matrix newMatrix = new Matrix(a.numRows, a.numColumns);
                for (int row = 0; row < a.numRows; row++)
                {
                    for (int column = 0; column < a.numColumns; column++)
                    {
                        newMatrix.matrix[row, column] = a.matrix[row, column] - b.matrix[row, column];
                    }
                }

                return newMatrix;
            }

            return new Matrix(0, 0);
        }

        public static Matrix DivideByConstant(Matrix a, double c)
        {
            for (int row = 0; row < a.numRows; row++)
            {
                for (int column = 0; column < a.numColumns; column++)
                {
                    a.matrix[row, column] /= c;
                }
            }
            return a;
        }

        public static Matrix Square(Matrix a)
        {
            for (int row = 0; row < a.numRows; row++)
            {
                for (int column = 0; column < a.numColumns; column++)
                {
                    a.matrix[row, column] = Math.Pow(a.matrix[row, column], 2);
                }
            }
            return a;
        }
        public static Matrix Multiply(Matrix a, Matrix b)
        {
            if (a.numColumns == b.numRows)
            {
                Matrix newMatrix = new Matrix(a.numRows, b.numColumns);
                double[,] rawA = a.matrix, rawB = b.matrix, rawNew = newMatrix.matrix;

                for (int rowIdx = 0; rowIdx < a.numRows; rowIdx++)
                {
                    for (int columnIdx = 0; columnIdx < b.numColumns; columnIdx++)
                    {
                        double[] row = a.GetRow(rowIdx);
                        double[] column = b.GetColumn(columnIdx);

                        for (int i = 0; i < row.Length; i++)
                        {
                            rawNew[rowIdx, columnIdx] += row[i] * column[i];
                        }
                    }
                }

                return newMatrix;
            }

            return new Matrix(0, 0);
        }

        public static Matrix ApplyActivation(Matrix a)
        {
            for (int row = 0; row < a.matrix.GetLength(0); row++)
            {
                for (int column = 0; column < a.matrix.GetLength(1); column++)
                {
                    double x = a.matrix[row, column];
                    a.matrix[row, column] = Math.Tanh(x);
                }
            }

            return a;
        }

        public static Matrix ApplyDerivativeActivation(Matrix a)
        {
            for (int row = 0; row < a.matrix.GetLength(0); row++)
            {
                for (int column = 0; column < a.matrix.GetLength(1); column++)
                {
                    double x = a.matrix[row, column];
                    a.matrix[row, column] = (1 - Math.Pow(Math.Tanh(x), 2));
                }
            }

            return a;
        }

        public static Matrix ApplySoftmax(Matrix outputs) //applies to vectors only
        {
            Matrix newMatrix = new Matrix(outputs.numRows, outputs.numColumns);
            double esum = 0;
            foreach (double x in outputs.matrix)
            {
                esum += Math.Exp(x);
            }

            for (int row = 0; row < newMatrix.numRows; row++)
            {
                newMatrix.matrix[row, 0] = Math.Exp(outputs.matrix[row, 0]) / esum;
            }

            return newMatrix;
        }

        public static Matrix ApplySoftmaxDerivative(Matrix outputs) //applies to vectors only
        {
            //Matrix newMatrix = new Matrix(outputs.numRows, outputs.numColumns);
            //double esum = 0;
            //foreach (double x in outputs.matrix)
            //{
            //    esum += Math.Exp(x);
            //}

            //for (int row = 0; row < newMatrix.numRows; row++)
            //{
            //    double sumMinusThis = esum - Math.Exp(outputs.matrix[row, 0]);
            //    newMatrix.matrix[row, 0] = Math.Exp(outputs.matrix[row, 0]) * sumMinusThis / Math.Pow(esum, 2); ;
            //}

            //return newMatrix;

            Matrix newMatrix = new Matrix(outputs.numRows, outputs.numColumns);
            Matrix softMax = Matrix.ApplySoftmax(outputs);
            for (int row = 0; row < newMatrix.numRows; row++)
            {
                newMatrix.matrix[row, 0] = softMax.matrix[row, 0] * (1 - softMax.matrix[row, 0]);
            }

            return newMatrix;
        }
    }
}
