#include"SLICAP.h"








int main() {

	SLICAP engine("examples/example.png");





























	std::cout << "initializing SLICAP" << std::endl;
	SLICAP::initialize("examples/example.png");

	std::cout << "running SLIC" << std::endl;
	SLIC(); //Simple Linear Iterative Clustering: an algorithm to oversegment the image into a field of reguarly sized clusters, fittingly called superpixels.

	//check result
	cv::Mat h_labels(SLICAP::labels.size(), SLICAP::labels.type());
	SLICAP::labels.download(h_labels);
	cv::Mat SLIC_result(h_labels.size(), CV_8UC3);

	for(int col = 0; col < h_labels.cols; col++){
		for(int row = 0; row < h_labels.rows; row++){

			int label = h_labels.at<int>(col, row);
			int label_range = label % (256 * 3);
			uchar label_r = label_range % 256;
			uchar label_g = (label_range - label_r) % 256;
			uchar label_b = (label_range - label_g - label_r) % 256;
			SLIC_result.at<cv::Vec3b>(col, row) = {label_b, label_g, label_r};

		}
	}

	PRINT_MAT(SLICAP::source);
	PRINT_MAT(SLICAP::labels);
	PRINT_MAT(h_labels);
	PRINT_MAT(SLIC_result);
	


	cv::imshow("SLIC result", SLIC_result);
	cv::waitKey(0);



	std::cout << "running AP" << std::endl;
	AP(); //Affinity Propagation: a message passing algorithm which groups data points under their 'exemplars': suitable representatives of a large number of other data points.

	std::cout << "displaying result" << std::endl;
	SLICAP::display_result(); //we will use Affinity Propagation to associate the superpixels produced by SLIC into larger regions based on color distance, producing a segmentation of the original image. 

	return 0;
}