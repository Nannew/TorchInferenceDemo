#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <string.h>

#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkNiftiImageIO.h>


int main() {
	//Define inputs/outputs
	const std::string inputTorchModelname = "deep_pet_traced_model.pt";//Torch Script Module - traced model
	const std::string inputSinoVolname = "sinogramfile_4_T1.nii.gz";//Input Sino Volume - Height = 128, Width = 128, Slice = 181
	const std::string outputPETVolname = "output_imagefile_4_T1.nii.gz";//Output PET Img Volume

	// 1. Read sinogram volume with ITK
	// Define the image type
	using PixelType = float;
	constexpr unsigned int Dimension = 3;
	using ImageType = itk::Image<PixelType, Dimension>;

	itk::ImageFileReader<ImageType>::Pointer reader = itk::ImageFileReader<ImageType>::New();
	reader->SetFileName(inputSinoVolname);

	// Set the image IO to NiftiImageIO
	using NiftiImageIOType = itk::NiftiImageIO;
	NiftiImageIOType::Pointer niftiImageIO = NiftiImageIOType::New();
	reader->SetImageIO(niftiImageIO);

	try
	{
		reader->Update();
	}
	catch (itk::ExceptionObject &err)
	{
		std::cerr << "Exception caught: " << err << std::endl;
		return EXIT_FAILURE;
	}

	ImageType::Pointer image = reader->GetOutput();

	// Get the size of the image
	ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
	std::cout << "Image size: " << size << std::endl;

	// 2. Convert the image data to a PyTorch tensor
	// Assuming the input size is (128, 128) as expected by the model, we slice the last dimension (depth)
	int height = size[0];  // Y-dimension
	int width = size[1];   // X-dimension
	int depth = size[2];   // Z-dimension
	std::cout << "Image size: " << height << " x " << width << " x " << depth << std::endl;

	// Extract image data
	ImageType::PixelContainer::Pointer pixelContainer = image->GetPixelContainer();
	PixelType* pixelData = pixelContainer->GetBufferPointer();

	// 3. Load Torch Script Module - traced model
	//https://pytorch.org/tutorials/advanced/cpp_export.html
	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load()
		module = torch::jit::load(inputTorchModelname);
	}
	catch (const c10::Error& e) {
		std::cerr << "Error loading the model\n";
		return -1;
	}

	// Create an empty output image with the same size as the input image
	ImageType::Pointer outputImage = ImageType::New();
	ImageType::RegionType region;
	ImageType::IndexType start = { 0, 0, 0 };
	region.SetSize(size);
	region.SetIndex(start);
	outputImage->SetRegions(region);
	outputImage->Allocate();
	outputImage->FillBuffer(0.0);  // Initialize with zeros

	PixelType* outputPixelData = outputImage->GetPixelContainer()->GetBufferPointer();

	// 4. Iterate over each depth slice (Z-dimension) and copy output slice to PET image volume
	for (int z = 0; z < depth; ++z) {
		// Extract the 2D slice from the 3D image (height x width)
		// Remember: ITK stores images in row-major order (Y, X, Z)
		PixelType* slice_data = pixelData + z * height * width;

		// Convert this slice to a PyTorch tensor
		at::Tensor tensor_image = torch::from_blob(slice_data, { 1, 1, height, width }, torch::kFloat).clone(); // [batch_size, channels, height, width]

		// Run the model's forward pass on this slice
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(tensor_image);

		// Perform inference on the slice
		at::Tensor output = module.forward(inputs).toTensor();

		// Ensure output is in the correct format (i.e., [1, 1, height, width])
		output = output.squeeze().detach();  // Remove batch and channel dimensions

		// Copy the output slice back to the output 3D ITK image
		PixelType* output_slice_data = outputPixelData + z * height * width;
		std::memcpy(output_slice_data, output.data_ptr<PixelType>(), height * width * sizeof(PixelType));

		// Print the output tensor size for this slice
		std::cout << "Slice " << z << " processed - " << "output size: " << output.sizes() << std::endl;
	}

	// 5. Save the 3D output image to disk using ITK
	using WriterType = itk::ImageFileWriter<ImageType>;
	WriterType::Pointer writer = WriterType::New();
	writer->SetFileName(outputPETVolname);
	writer->SetInput(outputImage);

	try {
		writer->Update();
		std::cout << "Output image saved to " << outputPETVolname << std::endl;
	}
	catch (itk::ExceptionObject &ex) {
		std::cerr << "Error during writing: " << ex << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}