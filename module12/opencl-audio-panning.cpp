#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
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
    std::vector<cl_kernel> kernels;
    std::vector<cl_command_queue> queues;
    std::vector<cl_mem> buffers;
    int * inputOutput;

    int platform = DEFAULT_PLATFORM; 
    bool useMap  = DEFAULT_USE_MAP;

    std::cout << "Audio panning assignment by Tyler Feldman" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);
    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 
    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

	// Load program source file
    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

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

    // create buffers and sub-buffers
    inputOutput = new int[NUM_BUFFER_ELEMENTS];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
        inputOutput[i] = i;
    }

    // create a single buffer to cover all the input data
    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // now for all devices other than the first create a sub-buffer
	cl_buffer_region region = 
		{
			NUM_BUFFER_ELEMENTS * 0 * sizeof(int), 
			NUM_BUFFER_ELEMENTS * sizeof(int)
		};
	cl_mem buffer = clCreateSubBuffer(
		main_buffer,
		CL_MEM_READ_WRITE,
		CL_BUFFER_CREATE_TYPE_REGION,
		&region,
		&errNum);
	checkErr(errNum, "clCreateSubBuffer");

	buffers.push_back(buffer);

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
	queues.push_back(queue);

	cl_kernel kernel = clCreateKernel(
		program,
		"square",
		&errNum);
	checkErr(errNum, "clCreateKernel(square)");

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[0]);
	checkErr(errNum, "clSetKernelArg(square)");

	kernels.push_back(kernel);

	// Write input data
	errNum = clEnqueueWriteBuffer(
		queues[numDevices - 1],
		main_buffer,
		CL_TRUE,
		0,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput,
		0,
		NULL,
		NULL);

    std::vector<cl_event> events;
    // call kernel for each device
	cl_event event;
	size_t gWI = NUM_BUFFER_ELEMENTS;
	errNum = clEnqueueNDRangeKernel(
		queues[0], 
		kernels[0], 
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
		queues[numDevices - 1],
		main_buffer,
		CL_TRUE,
		0,
		sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
		(void*)inputOutput,
		0,
		NULL,
		NULL);

    // Display output in rows
	for (unsigned elems = 0 * NUM_BUFFER_ELEMENTS; elems < ((0+1) * NUM_BUFFER_ELEMENTS); elems++)
	{
		std::cout << " " << inputOutput[elems];
	}

	std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
