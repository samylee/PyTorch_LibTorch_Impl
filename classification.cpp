#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat pilResize(Mat &img, int size) {
	int imgWidth = img.cols;
	int imgHeight = img.rows;
	if ((imgWidth <= imgHeight && imgWidth == size) || (imgHeight <= imgWidth && imgHeight == size)) {
		return img;
	}
	Mat output;
	if (imgWidth < imgHeight) {
		int outWidth = size;
		int outHeight = int(size * imgHeight / (float)imgWidth);
		resize(img, output, Size(outWidth, outHeight));
	}
	else {
		int outHeight = size;
		int outWidth = int(size * imgWidth / (float)imgHeight);
		resize(img, output, Size(outWidth, outHeight));
	}

	return output;
}

Mat pilCropCenter(Mat &img, int output_size) {
	Rect imgRect;
	imgRect.x = int(round((img.cols - output_size) / 2.));
	imgRect.y = int(round((img.rows - output_size) / 2.));
	imgRect.width = output_size;
	imgRect.height = output_size;

	return img(imgRect).clone();
}

Mat setNorm(Mat &img) {
	Mat img_rgb;
	cvtColor(img, img_rgb, COLOR_RGB2BGR);

	Mat img_resize = pilResize(img_rgb, 256);
	Mat img_crop = pilCropCenter(img_resize, 224);

	Mat image_resized_float;
	img_crop.convertTo(image_resized_float, CV_32F, 1.0 / 255.0);

	return image_resized_float;
}

Mat setMean(Mat &image_resized_float) {
	vector<float> mean = { 0.485, 0.456, 0.406 };
	vector<float> std = { 0.229, 0.224, 0.225 };

	vector<Mat> image_resized_split;
	split(image_resized_float, image_resized_split);
	for (int ch = 0; ch < image_resized_split.size(); ch++) {
		image_resized_split[ch] -= mean[ch];
		image_resized_split[ch] /= std[ch];
	}
	Mat image_resized_merge;
	merge(image_resized_split, image_resized_merge);

	return image_resized_merge;
}

int main() {
	torch::DeviceType device_type;
	if (torch::cuda::is_available()) {
		std::cout << "CUDA available! Test on GPU." << std::endl;
		device_type = torch::kCUDA;
	}
	else {
		std::cout << "Test on CPU." << std::endl;
		device_type = torch::kCPU;
	}
	torch::Device device(device_type);

	// Deserialize the ScriptModule from a file using torch::jit::load().
	torch::jit::script::Module model = torch::jit::load("model_cpu.pt");
	model.to(device);

	vector<string> classes = { "cat","dog" };

	string test_path = "val/dog/";
	vector<string> img_paths;
	glob(test_path, img_paths);

	int truth_count = 0;

	for (int i = 0; i < img_paths.size(); i++) {
		Mat img = imread(img_paths[i]);

		clock_t start_t = clock();

		//norm
		Mat image_resized_float = setNorm(img);
		//mean
		Mat image_resized_merge = setMean(image_resized_float);

		auto img_tensor = torch::from_blob(image_resized_merge.data, { 224, 224, 3 }, torch::kFloat32);
		auto img_tensor_ = torch::unsqueeze(img_tensor, 0);
		img_tensor_ = img_tensor_.permute({ 0, 3, 1, 2 });

		// Create a vector of inputs.
		vector<torch::jit::IValue> inputs;
		inputs.push_back(img_tensor_.to(device));

		torch::Tensor prob = model.forward(inputs).toTensor();
		torch::Tensor output = torch::softmax(prob, 1);
		auto predict = torch::max(output, 1);

		//cout << "cost time:" << clock() - start_t << endl;

		cout << img_paths[i] << "\t";
		cout << "class: " << classes[get<1>(predict).item<int>()] <<
			", prob: " << get<0>(predict).item<float>() << endl;

		if (get<1>(predict).item<int>() == 1) {
			truth_count++;
		}
	}

	cout << truth_count << "/" << img_paths.size() << endl;
	system("pause");

	return 0;
}