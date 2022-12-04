#include"core/SLICAP.h"








int main() {

	std::cout << "loading source" << std::endl;
	h_Mat source = cv::imread("examples/example.png", cv::IMREAD_COLOR);

	std::cout << "initializing engine" << std::endl;
	SLICAP engine(source);

	std::cout << "running SLIC" << std::endl;
	engine.run_SLIC();

	std::cout << "displaying labels" << std::endl;
	engine.display_SLIC_result();

	std::cout << "running AP" << std::endl;
	engine.run_AP();

	std::cout << "displaying result" << std::endl;
	engine.display_AP_result();

	return 0;
}