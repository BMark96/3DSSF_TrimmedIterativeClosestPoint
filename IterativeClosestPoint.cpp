#include <iostream>
#include <chrono>
#include <random>
#include <Eigen/Dense>
#include <nanoflann.hpp>
#include "happly.h"

typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> kd_tree_t;

// Importing PLY file into Eigen Matrix
Eigen::MatrixXd readPLY(std::string fileName) {
	happly::PLYData PLYData(fileName);

	// Read PLY file
	std::vector<std::array<double, 3>> positions = PLYData.getVertexPositions();
	
	// Filling matrix
	Eigen::MatrixXd pointCloud(positions.size(), 3);
	for (int i = 0; i < pointCloud.rows(); ++i) {
		for (int j = 0; j < pointCloud.cols(); ++j) {
			pointCloud(i, j) = positions[i][j];
		}
	}
	return pointCloud;
}

// Exporting pointcloud from Eigen matrix to PLY file
void writePLY(Eigen::MatrixXd pointCloud, std::string fileName) {
	std::vector<std::array<double, 3>> positions;

	// Converting Eigen matrix to std vector of arrays
	int nPoints = pointCloud.rows();
	for (int i = 0; i < nPoints; ++i) {
		std::array<double, 3> point = { pointCloud(i, 0), pointCloud(i, 1), pointCloud(i, 2) };
		positions.push_back(point);
	}

	// Write into PLY file
	happly::PLYData PLYData;
	PLYData.addVertexPositions(positions);
	PLYData.write(fileName, happly::DataFormat::ASCII);
}

// Add gaussian noise to points stored in Eigen matrix (if stdDev=0 => no noise added)
Eigen::MatrixXd addNoise(Eigen::MatrixXd original, double stdDev) {
	int nPoints = original.rows();

	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0, stdDev);

	// Generating and adding noise
	Eigen::MatrixXd withNoise(nPoints, 3);
	for (int i = 0; i < nPoints; ++i) {
		for (int j = 0; j < 3; ++j) {
			withNoise(i, j) = original(i, j) + distribution(generator);
		}
	}
	return withNoise;
}

// Finding nearest neighbor for all data points in the set of model points using nanoflann
std::vector<std::tuple<int, int, double>> findNNs(Eigen::MatrixXd modelPointCloud, Eigen::MatrixXd dataPointCloud) {
	int nDataPoints = dataPointCloud.rows();

	// Building KD-tree for model
	kd_tree_t modelKDTree(3, std::cref(modelPointCloud), 10);
	modelKDTree.index->buildIndex();

	// Finding neirest neighbor and square distance for all data points
	std::vector<std::tuple<int, int, double>> NNs;
	for (int i = 0; i < nDataPoints; ++i) {
		// Query point
		std::vector<double> dataPoint{dataPointCloud(i, 0), dataPointCloud(i, 1), dataPointCloud(i, 2)};
		
		// Initialize search
		std::vector<size_t> NNindex(1);
		std::vector<double> sqrDistance(1);
		nanoflann::KNNResultSet<double> result(1);
		result.init(&NNindex[0], &sqrDistance[0]);

		// Search
		modelKDTree.index->findNeighbors(result, &dataPoint[0], nanoflann::SearchParams(10));

		// Creating tuple of matched indices and square distance
		std::tuple<int, int, double> NN = std::make_tuple(i, NNindex[0], sqrDistance[0]);
		NNs.push_back(NN);
	}
	return NNs;
}

// Compare function for vector of tuples containing matched indices and square distances (compare square distances)
bool sortBySqrDist(const std::tuple<int, int, double>& a, const std::tuple<int, int, double>& b)
{
	return (std::get<2>(a) < std::get<2>(b));
}

// Calculating sum of square distances in a vector of tuples containing matched indices and square distances
double sumOfSquares(std::vector<std::tuple<int, int, double>> NNs) {
	size_t nElements = NNs.size();
	
	double sum = 0;
	for (int i = 0; i < nElements; ++i) {
		sum += std::get<2>(NNs[i]);
	}
	return sum;
}

// Compare function for eigenvalues (we only take real parts into account because N has only real eigenvalues)
bool compareComplex(std::complex<double> a, std::complex<double> b) {
	return real(a) < real(b);
}

// Trim NNs (keeping index-pairs corresponding to smallest square errors)
std::vector<std::tuple<int, int, double>> trimNNs(std::vector<std::tuple<int, int, double>> NNs, int nPointsToKeep) {
	sort(NNs.begin(), NNs.end(), sortBySqrDist);
	std::vector<std::tuple<int, int, double>>::iterator deleteFrom = NNs.begin() + nPointsToKeep;
	std::vector<std::tuple<int, int, double>>::iterator deleteTo = NNs.end();
	NNs.erase(deleteFrom, deleteTo);

	return NNs;
}

// Calculating aligmnent using unit quaternions (As explained here: http://www.sci.utah.edu/~shireen/pdfs/tutorials/Elhabian_ICP09.pdf)
std::pair<Eigen::MatrixXd, Eigen::VectorXd>  calculateAlignment(Eigen::MatrixXd modelForAlignment, Eigen::MatrixXd dataForAlignment) {
	// Substract mean from data and model
	Eigen::VectorXd dataMean = dataForAlignment.colwise().mean();
	Eigen::VectorXd modelMean = modelForAlignment.colwise().mean();
	dataForAlignment.rowwise() -= dataMean.transpose();
	modelForAlignment.rowwise() -= modelMean.transpose();

	// Create N matrix
	Eigen::MatrixXd S = dataForAlignment.transpose() * modelForAlignment;
	Eigen::MatrixXd N(4, 4);
	N << S(0, 0) + S(1, 1) + S(2, 2), S(1, 2) - S(2, 1), -S(0, 2) + S(2, 0), S(0, 1) - S(1, 0),
		-S(2, 1) + S(1, 2), S(0, 0) - S(2, 2) - S(1, 1), S(0, 1) + S(1, 0), S(0, 2) + S(2, 0),
		S(2, 0) - S(0, 2), S(1, 0) + S(0, 1), S(1, 1) - S(2, 2) - S(0, 0), S(1, 2) + S(2, 1),
		-S(1, 0) + S(0, 1), S(2, 0) + S(0, 2), S(2, 1) + S(1, 2), S(2, 2) - S(1, 1) - S(0, 0);

	// Searching maximum eigenvalue of N matrix
	Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(N);
	auto eigenValues = eigenSolver.eigenvalues();
	auto eigenVectors = eigenSolver.eigenvectors();
	auto maxEigenValue = max_element(eigenValues.begin(), eigenValues.end(), compareComplex);
	int maxEigenValueIndex = std::distance(eigenValues.begin(), maxEigenValue);

	// Elements of unit quaternion (elements of eigenvector corresponding to maximum eigenvalue)
	double q0 = real(eigenVectors(0, maxEigenValueIndex));
	double qX = real(eigenVectors(1, maxEigenValueIndex));
	double qY = real(eigenVectors(2, maxEigenValueIndex));
	double qZ = real(eigenVectors(3, maxEigenValueIndex));

	// Creating Q and QBar matrices
	Eigen::MatrixXd Q(4, 4);
	Q << q0, -qX, -qY, -qZ,
		qX, q0, -qZ, qY,
		qY, qZ, q0, -qX,
		qZ, -qY, qX, q0;
	Eigen::MatrixXd QBar(4, 4);
	QBar << q0, -qX, -qY, -qZ,
		qX, q0, qZ, -qY,
		qY, -qZ, q0, qX,
		qZ, qY, -qX, q0;

	// Calculating R
	Eigen::MatrixXd R = (QBar.transpose() * Q).block<3, 3>(1, 1);

	// Calculating t
	Eigen::VectorXd t = modelMean - R * dataMean;

	return std::make_pair(R, t);
}

// Iterative closest point algorithm using nearest neighbor search for finding correspondences and unit quaternions for finding alignments
Eigen::MatrixXd ICP(Eigen::MatrixXd model, Eigen::MatrixXd data, double overlap, int nIterMax, double errorThreshold, double errorChangeThreshold) {
	int nDataPoints = data.rows();
	Eigen::MatrixXd dataTransformed = data;
	int nPointsToAlign = nDataPoints * overlap;
	double errorOld = 1000000; 
	std::vector<double> errors;

	for (int i = 1; i <= nIterMax; ++i) {
		// Find nearest neighbor for all data points in the set of model points
		std::vector<std::tuple<int, int, double>> NNs = findNNs(model, dataTransformed);

		// If overlap < 1 we use tr-ICP, we trim NNs
		if (overlap < 1) {
			NNs = trimNNs(NNs, nPointsToAlign);
		}

		// Calculating MSE and error change before alignment
		double errorNew = sumOfSquares(NNs) / nPointsToAlign;
		double errorChange = abs(errorNew - errorOld);
		errors.push_back(errorNew);

		std::cout << i << ": " << errorNew << std::endl;

		// Checking exit criteria
		if (errorNew < errorThreshold || errorChange < errorChangeThreshold) {
			break;
		}
		errorOld = errorNew;

		// Reorder data for alignment (pairs with same indices)
		Eigen::MatrixXd dataForAlignment(nPointsToAlign, 3);
		Eigen::MatrixXd modelForAlignment(nPointsToAlign, 3);
		for (int i = 0; i < nPointsToAlign; ++i) {
			dataForAlignment.row(i) = dataTransformed.row(std::get<0>(NNs[i]));
			modelForAlignment.row(i) = model.row(std::get<1>(NNs[i]));
		}

		// Calculate alignment
		std::pair<Eigen::MatrixXd, Eigen::VectorXd> alignment = calculateAlignment(modelForAlignment, dataForAlignment);
		Eigen::MatrixXd R = alignment.first;
		Eigen::VectorXd t = alignment.second;

		// Transform data
		dataTransformed = (dataTransformed * R.transpose()).rowwise() + t.transpose();
	}

	// Print errors (for plotting)
	for (std::vector<double>::iterator i = errors.begin(); i != errors.end(); ++i)
		std::cout << *i << ' ';
	std::cout << std::endl;

	return dataTransformed;
}

int main(int argc, char** argv)
{
	// Checking number of arguments
	if (argc != 9) {
		std::cout << "8 command line arguments needed: Model path, Data path, Transformed data path, Standard deviation of Gaussian noise, Overlap, Number of max ICP iterations, Error threshold, Error change threshold" << std::endl;
		return -1;
	}

	// Reading arguments
	std::string modelPath = argv[1];
	std::string dataPath = argv[2];
	std::string transformedDataPath = argv[3];
	double noiseStdDev = atof(argv[4]);
	double overlap = atof(argv[5]);
	int nIterMax = atoi(argv[6]);
	double errorThreshold = atof(argv[7]);
	double errorChangeThreshold = atof(argv[8]);

	// Reading point clouds into Eigen matrices
	Eigen::MatrixXd model = readPLY(modelPath);
	Eigen::MatrixXd data = readPLY(dataPath);

	// Add noise to model and data
	model = addNoise(model, noiseStdDev);
	data = addNoise(data, noiseStdDev);

	// Align data matrix with ICP and measure runtime
	auto start = std::chrono::high_resolution_clock::now();
	Eigen::MatrixXd transformedData = ICP(model, data, overlap, nIterMax, errorThreshold, errorChangeThreshold);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Runtime of ICP: " << duration.count() << "ms." << std::endl;

	// Export aligned data matrix
	writePLY(transformedData, transformedDataPath);
}