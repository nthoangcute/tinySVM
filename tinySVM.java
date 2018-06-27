import java.io.*;
import java.nio.file.Files;
import java.util.*;

public class tinySVM {

	public String trainFile = "/home/vietbt/java/mnist_digits_train.txt";
	public String testFile = "/home/vietbt/java/mnist_digits_test.txt";
	public int outputSize = 10;
	public double e = 1000;
	public double C = 10.0;
	public double sigma = 1.0;
	public double tol = 0.00001;
	public int maxiter = 10000;
	public int numpasses = 100;
	public int batchSize = 45000;
	public int stride = 9;
	public int padding = 6;
	public int orient = 6;
	private double[][][] trainData;
	private double[] trainLabel;
	private double[][][] testData;
	private double[] testLabel;
	private double[][] dx = { { -1, 0, 1 } };
	private double[][] dy = { { -1 }, { 0 }, { 1 } };
	private double[][] listLabel;
	private double[][] data;
	private double[][] test;
	private double[][] kernelData;
	private double[][] kernelTest;
	private int N;
	private int D;
	private double[] bias;
	private double[][] w;
	private double[][] alpha;
	private int testNumb = 1;

	public static void main(String[] args) throws Exception {
		new tinySVM().run();
	}

	public void run() throws Exception {
		System.out.print("Preparing ... ");
		readTrainFile(trainFile);
		readTestFile(testFile);
		N = trainData.length;
		D = getHOG(trainData[0]).length;
		data = new double[N][D];
		test = new double[testData.length][D];
		kernelData = new double[N][batchSize];
		kernelTest = new double[testData.length][batchSize];
		for (int i = 0; i < N; i++)
			data[i] = getHOG(trainData[i]);
		for (int i = 0; i < testData.length; i++)
			test[i] = getHOG(testData[i]);
		for (int i = 0; i < N; i++)
			for (int j = 0; j < batchSize; j++)
				kernelData[i][j] = kernel(data[i], data[j]);
		for (int i = 0; i < testData.length; i++)
			for (int j = 0; j < batchSize; j++)
				kernelTest[i][j] = kernel(test[i], data[j]);
		w = new double[outputSize - 1][D];
		alpha = new double[outputSize - 1][N];
		bias = new double[outputSize - 1];
		listLabel = new double[outputSize - 1][N];
		for (int i = 0; i < outputSize - 1; i++)
			for (int j = 0; j < N; j++)
				listLabel[i][j] = trainLabel[j] <= i ? -1 : 1;
		System.out.println("Done!\nStarting ... ");
		new Thread(() -> {SMO(0);}).start();
		new Thread(() -> {SMO(1);}).start();
		new Thread(() -> {SMO(2);}).start();
		new Thread(() -> {SMO(3);}).start();
		new Thread(() -> {SMO(4);}).start();
		new Thread(() -> {SMO(5);}).start();
		new Thread(() -> {SMO(6);}).start();
		new Thread(() -> {SMO(7);}).start();
		new Thread(() -> {SMO(8);}).start();
		new Thread(() -> {
			while (true) {
				print("Test " + testNumb++ + "\n");
				int correct = 0;
				int[][] map = new int[10][10];
				for (int k = 0; k < testData.length; k++) {
					if (k % 100 == 0) print(".");
					int ans = outputSize - 1;
					for (int i = 0; i < outputSize - 1; i++)
						if (funct(k, i) < 0) {
							ans = i;
							break;
						}
					if (ans == testLabel[k]) correct++;
					map[ans][(int) testLabel[k]]++;
				}
				print("\nAccuracy = " + 100.0 * correct / testData.length + "%\n");
				for (int[] x : map) {
					for (int y : x) print(y + "\t");
					print("\n");
				}
			}
		}).start();
	}

	public void readTrainFile(String filePath) throws Exception {
		List<String> lines = Files.readAllLines(new File(filePath).toPath());
		trainData = new double[lines.size()][28][28];
		trainLabel = new double[lines.size()];
		readFile(lines, trainData, trainLabel);
	}

	public void readTestFile(String filePath) throws Exception {
		List<String> lines = Files.readAllLines(new File(filePath).toPath());
		testData = new double[lines.size()][28][28];
		testLabel = new double[lines.size()];
		readFile(lines, testData, testLabel);
	}

	public void readFile(List<String> lines, double[][][] data, double[] label) {
		int index = 0;
		for (String line : lines) {
			double[] arr = Arrays.stream(line.split("\\|")).mapToDouble(Double::parseDouble).toArray();
			int k = 0;
			for (int i = 0; i < 28; i++)
				for (int j = 0; j < 28; j++)
					data[index][i][j] = arr[k++];
			label[index++] = arr[k];
		}
	}

	public void print(String s) {
		try (PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("log_svm.txt", true)))) {
			System.out.print(s);
			out.print(s);
		} catch (Exception x) {}
	}

	public double funct(int x, int k) {
		double f = bias[k];
		for (int i = 0; i < N; i++)
			f += alpha[k][i] * listLabel[k][i] * kernelt(x, i);
		return f;
	}

	public double func(double[] x, int k) {
		double f = bias[k];
		for (int i = 0; i < N; i++)
			f += alpha[k][i] * listLabel[k][i] * kernel(x, data[i]);
		return f;
	}

	public double func(int x, int k) {
		double f = bias[k];
		for (int i = 0; i < N; i++)
			f += alpha[k][i] * listLabel[k][i] * kernel(x, i);
		return f;
	}

	public double kernel(int i, int j) {
		if (i >= batchSize && j >= batchSize) return kernel(data[i], data[j]);
		else return i > j ? kernelData[i][j] : kernelData[j][i];
	}

	public double kernelt(int i, int j) {
		return j >= batchSize ? kernel(test[i], data[j]) : kernelTest[i][j];
	}

	public double kernel(double[] a, double[] b) {
		double sum = 0;
		for (int i = 0; i < a.length; i++)
			sum += (a[i] - b[i]) * (a[i] - b[i]);
		return Math.exp(-sum / (2.0 * sigma * sigma));
	}

	public void SMO(int k) {
		int iter = 0, passes = 0;
		while (passes < numpasses && iter < maxiter) {
			int alphaChanged = 0;
			for (int i = 0; i < N; i++) {
				double Ei = func(i, k) - listLabel[k][i];
				if ((listLabel[k][i] * Ei < -tol && alpha[k][i] < C)
						|| (listLabel[k][i] * Ei > tol && alpha[k][i] > 0)) {
					int j = i;
					while (j == i) j = new Random().nextInt(N);
					double Ej = func(j, k) - listLabel[k][j];
					double L = listLabel[k][i] == listLabel[k][j] ? Math.max(0, alpha[k][i] + alpha[k][j] - C)
							: Math.max(0, alpha[k][j] - alpha[k][i]);
					double H = listLabel[k][i] == listLabel[k][j] ? Math.min(C, alpha[k][i] + alpha[k][j])
							: Math.min(C, C + alpha[k][j] - alpha[k][i]);
					if (Math.abs(L - H) < 1e-4) continue;
					double eta = 2 * kernel(i, j) - kernel(i, i) - kernel(j, j);
					if (eta >= 0) continue;
					double newaj = alpha[k][j] - listLabel[k][j] * (Ei - Ej) / eta;
					if (newaj > H) newaj = H;
					if (newaj < L) newaj = L;
					if (Math.abs(alpha[k][j] - newaj) < 1e-4) continue;
					alpha[k][j] = newaj;
					double newai = alpha[k][i] + listLabel[k][i] * listLabel[k][j] * (alpha[k][j] - newaj);
					alpha[k][i] = newai;
					double b1 = bias[k] - Ei - listLabel[k][i] * (newai - alpha[k][i]) * kernel(i, i)
							- listLabel[k][j] * (newaj - alpha[k][j]) * kernel(i, j);
					double b2 = bias[k] - Ej - listLabel[k][i] * (newai - alpha[k][i]) * kernel(i, j)
							- listLabel[k][j] * (newaj - alpha[k][j]) * kernel(j, j);
					bias[k] = 0.5 * (b1 + b2);
					if (newai > 0 && newai < C) bias[k] = b1;
					if (newaj > 0 && newaj < C) bias[k] = b2;
					alphaChanged++;
				}
			}
			iter++;
			passes += alphaChanged == 0 ? 1 : 0;
		}
		for (int j = 0; j < D; j++)
			for (int i = 0; i < N; i++)
				w[k][j] += alpha[k][i] * listLabel[k][i] * data[i][j];
	}

	public double[] getHOG(double[][] image) {
		double[][] x = conv(image, dx);
		double[][] y = conv(image, dy);
		int m = x.length;
		int n = y[0].length;
		double[][] g = new double[m][n];
		int[][] a = new int[m][n];
		for (int i = 0; i < m; i++)
			for (int j = 0; j < n; j++) {
				double u = 0, v = 0;
				if (j > 0 && j < n - 1) u = x[i][j - 1];
				if (i > 0 && i < m - 1) v = y[i - 1][j];
				g[i][j] = Math.sqrt(u * u + v * v);
				a[i][j] = ((int) (orient * Math.abs(Math.atan2(u, v)) / Math.PI)) % orient;
			}
		int size = (m - stride) / padding + 1;
		double[][][] d = new double[size][size][orient];
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				for (int si = 0; si < stride; si++)
					for (int sj = 0; sj < stride; sj++) {
						int px = i * padding + si;
						int py = j * padding + sj;
						d[i][j][a[px][py]] += g[px][py];
					}
		for (int i = 0; i < size; i++) {
			double[] sum = new double[orient];
			for (int j = 0; j < size; j++)
				for (int k = 0; k < orient; k++)
					sum[k] += d[i][j][k];
			for (int j = 0; j < size; j++)
				for (int k = 0; k < orient; k++)
					d[i][j][k] /= Math.sqrt(sum[k] * sum[k] + e * e);
		}
		double[] output = new double[size * size * orient];
		int index = 0;
		for (double[][] i : d)
			for (double[] j : i)
				for (double k : j)
					output[index++] = k;
		return output;
	}

	public double[][] conv(double[][] m, double[][] f) {
		int x = m.length - f.length + 1;
		int y = m[0].length - f[0].length + 1;
		double[][] result = new double[x][y];
		for (int i = 0; i < x; i++)
			for (int j = 0; j < y; j++) {
				double sum = 0.0;
				for (int p = 0; p < f.length; p++)
					for (int q = 0; q < f[0].length; q++)
						sum += m[i + p][j + q] * f[p][q];
				result[i][j] = sum;
			}
		return result;
	}
}
