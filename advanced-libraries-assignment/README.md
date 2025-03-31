# Advanced Libraries Assignment: Computing PI with cuRAND and Thrust
Author: Tyler Feldman / tfeldma7

## Description
This is a program utilizing cuRAND and thrust to compute an estimate of pi.

## Prerequisites
- Cuda installed
- make installed

## Building and Running
```bash
# Build
make clean && make

# Run with default number of points to sample
./advanced_lib_assignment 

# Run and specify number of points to sample
./advanced_lib_assignment 1234

# Profile
make profile
```

## Results
Below is a sample of results from profiling the code with `make profile`. As the number of points increases, the estimate of pi becomes more accurate and the time required to run the code increases as well.

| Number of Points | Estimate of Pi | Milliseconds Elapsed |
|-----------------|---------------|----------------------|
| 128            | 3.093750      | 7.788192            |
| 512            | 3.179688      | 7.623872            |
| 4,096          | 3.117188      | 7.504512            |
| 32,768        | 3.133667      | 7.373824            |
| 262,144       | 3.137329      | 7.582816            |
| 2,097,152     | 3.141222      | 7.855296            |
| 16,777,216    | 3.141363      | 9.812320            |
| 134,217,728   | 3.141704      | 26.408863           |

## Goals
My goal with this project was to utilize thrust and cuRAND to compute an estimate of pi with varying numbers of points to sample. I wanted to use cuRAND to sample the points and then use thrust to compute the ratio of points within a unit circle to the total number of points. I then wanted to use that ratio to compute an estimate of pi. This workflow for estimating pi is inspired by existing guides and descriptions such as the one on [geeksforgeeks.org](https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/).

Specifically, I used cuRAND to uniformly sample x and y points in a square of area 1 from 0-1 in the x axis and 0-1 in the y axis. cuRAND allowed me to parallelize this uniform sampling and perform the sampling on device. Next, I used thrust to compute the number of x,y points that had a magnitude less than 1 to count how many points satisfied this condition. I then used this to compute an estimate of pi!

## Challenges
During this assignment, I had lots of issues with including and linking the CUDA libraries. I initially tried to do an image processing assignment using FreeImage and NPP. However, I ran into many issues linking the FreeImage library. I kept getting undefined reference errors, indicating that the linker could not find the implementation of defined functions in the header files, despite linking the location of the static library provided in module8. I then tried working with nvGraph but discovered that nvGraph was removed in CUDA 11.0+ and moved to the RAPIDS project.

I also had issues reading and understanding documentation, particularly the documentation for thrust. I noticed that some functions in thrust had very short descriptions (e.g. [make_zip_iterator](https://nvidia.github.io/cccl/thrust/api/function_group__fancyiterator_1ga6727929f3d9d7fc699278849d9dda344.html#thrust-make-zip-iterator)) and had types in the function definition that could be confusing or unclear. Navigating the documentation was generally a challenge as someone with limited experience using the library and as someone who had only read the introduction and examples that the thrust documenation provides.

## Triumphs
Thrust was very easy to get working as all of the header files were located in my cuda/include directory. Using cuRAND was fairly simple, especially with the examples from previous modules as a reference point. I was quite pleased with how easy it was to load the device pointer I used for cuRAND into a thrust device pointer. Getting the custom operator to work with thrust was a big triumph, as there weren't any examples showing how to incorporate that with tuples and zipped iterators.

To overcome the issues with navigating documentation, I found that chatbots could point me in the right direction in terms of what functions to look at to achieve my goals. This way, I could specify my goals and get recommendations for functions to look at, as opposed to having to look through all of the functions in a particular section of the documentation until I found one or multiple that I could utilize.
