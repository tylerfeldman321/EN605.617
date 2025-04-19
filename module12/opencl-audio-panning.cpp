#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sndfile.h>

#include "info.hpp"

#define DEFAULT_PLATFORM 0

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


bool write_wav_file(const std::string& filename, const std::vector<short>& data, int sample_rate, int channels) {
    SF_INFO sfinfo;
    sfinfo.frames = data.size() / channels;
    sfinfo.samplerate = sample_rate;
    sfinfo.channels = channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outfile = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << sf_strerror(NULL) << std::endl;
        return false;
    }

    sf_count_t count = sf_write_short(outfile, data.data(), data.size());
    if (count != static_cast<sf_count_t>(data.size())) {
        std::cerr << "Error writing samples to file." << std::endl;
        sf_close(outfile);
        return false;
    }

    sf_close(outfile);
    return true;
}


///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;

    int platform = DEFAULT_PLATFORM; 

    // Get wav file as an argument
	if (argc < 2)
	{	puts ("Apply panning to an input stereo wav file. ") ;
		puts ("    Usage : generate <wav_file>\n") ;
		exit (1) ;
    };

    // Load audio data
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(argv[1], SFM_READ, &sfinfo);
    if (!sndfile) {
        printf("Error opening input file %s: %s\n", argv[1], (NULL));
        return 1;
    }
    std::vector<short> samples(sfinfo.frames * sfinfo.channels);
    int original_format = sfinfo.format;
    sf_readf_short(sndfile, samples.data(), sfinfo.frames);
    sf_close(sndfile);
    std::cout << "Loaded " << samples.size() << " samples.\n";
    std::cout << "Loaded samples from " << sfinfo.channels << " channels\n";
    std::cout << "Sample data: ";
    for (int i = 0; i < 1000; i++) {
        std::cout << " " << samples[i];
    }
    std::cout << "\n";

    std::cout << "Audio panning assignment by Tyler Feldman" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), "clGetPlatformIDs"); 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);
    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

	// Load program source file
    std::ifstream srcFile("pan-audio.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading pan-audio.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }
    std::cout << "Built program successfully!\n";

    // Create a buffer for our audio data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(short) * samples.size(),
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // Create command queues
	InfoDevice<cl_device_type>::display(
		deviceIDs[0], 
		CL_DEVICE_TYPE, 
		"CL_DEVICE_TYPE");

	cl_command_queue queue = 
		clCreateCommandQueue(
			context,
			deviceIDs[0],
			0,
			&errNum);
	checkErr(errNum, "clCreateCommandQueue");

	cl_kernel kernel = clCreateKernel(
		program,
		"pan_audio_2channel",
		&errNum);
	checkErr(errNum, "clCreateKernel(pan_audio_2channel)");

    // Set arguments
	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&main_buffer);
	checkErr(errNum, "clSetKernelArg(pan_audio_2channel)");
    const int num_frames = sfinfo.frames;
    std::cout << "Number of frames: " << num_frames << "\n";
    errNum = clSetKernelArg(kernel, 1, sizeof(int), (void *)&num_frames);
	checkErr(errNum, "clSetKernelArg(pan_audio_2channel)");
    const float pan = 0.9;
    errNum = clSetKernelArg(kernel, 2, sizeof(float), (void *)&pan);
	checkErr(errNum, "clSetKernelArg(pan_audio_2channel)");

	// Write input data
	errNum = clEnqueueWriteBuffer(
		queue,
		main_buffer,
		CL_TRUE,
		0,
		sizeof(short) * samples.size(),
		(void*)&samples[0],
		0,
		NULL,
		NULL);

    std::vector<cl_event> events;
    
    // Call the kernel
	cl_event event;
	size_t gWI = samples.size();
	errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel,
		1, 
		NULL,
		(const size_t*)&gWI, 
		(const size_t*)NULL, 
		0, 
		0, 
		&event);

	events.push_back(event);

    // Technically don't need this as we are doing a blocking read
    // with in-order queue.
    clWaitForEvents(events.size(), &events[0]);

	// Read back computed data
	clEnqueueReadBuffer(
		queue,
		main_buffer,
		CL_TRUE,
		0,
		sizeof(short) * samples.size(),
		(void*)&samples[0],
		0,
		NULL,
		NULL);

    std::cout << "Sample data: ";
    for (int i = 0; i < 1000; i++) {
        std::cout << " " << samples[i];
    }
    std::cout << "\n";
    std::cout << "Have " << samples.size() << " samples.\n";
	std::cout << std::endl;

    // sfinfo.format = original_format;
    // sfinfo.frames = 0;
    // const char* output_file = "output.wav";
    // printf("Frames: %lld, Channels: %d, Format: 0x%x\n", 
    //     sfinfo.frames, sfinfo.channels, sfinfo.format);
    // SNDFILE* outFile = sf_open(output_file, SFM_WRITE, &sfinfo);
    // if (!outFile) {
    //     printf("Error opening output file %s: %s\n", output_file, sf_strerror(NULL));
    //     return 1;
    // }
    // sf_writef_short(outFile, samples.data(), sfinfo.frames);
    // sf_close(outFile);
    // std::cout << "Wrote data to: " << output_file << "\n";
    if (write_wav_file("output.wav", samples, sfinfo.samplerate, 2)) {
        std::cout << "Stereo WAV file written successfully!" << std::endl;
    }

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
