# Module 12 Assignment - Audio Panning
Tyler Feldman

## Prequisites
- Ubuntu 22.04
- OpenCL installed
- Install libsndfile:
```bash
sudo apt-get update
sudo apt-get install libsndfile1-dev
```

## Building and Running
```bash
make

# Format for running the binary
./audio-panning.exe <audio-file.wav> <panning-level>

# Example call
./audio-panning.exe M1F1-uint8-AFsp.wav 0.8
```

## Resources
- Sample audio file taken from here: https://mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Samples.html

# Written Report
Beyond just the code and proof of its execution, create a one-page (font 12 or 14, double spaced, etc. and it can be a little shorter or longer) to discuss what your goals were and any challenges and triumphs that you had. 

## Goals
My goal with this assignment was to leverage OpenCL to create a useful application that leverages vectors, buffers, and reads and writes data to external files. Much of the code we look at as examples (like hello world style programs) performs trivial tasks. I wanted this assignment to provide some value to me. Specifically, I wanted to do audio processing, since that is a good use case for parallelization. I chose to write an OpenCL program to perform audio file panning using short2 vectors to represent the data, which involves adjusting the gain levels of a left vs. right channel.

## Challenges
During the development and testing of this program, I ran into several challenges. A couple were related to audio processing. I first had to find a library I could use to read in stereo audio data since that was necessary for panning. Installing the library and compiling a program with it was not too difficult. Initially when I got the program running, I was using a different sampling rate when saving the output wav file, create a speedup in the audio data.

Some OpenCL challenges I ran into were having the buffer in my OpenCL kernel as read-only, which required modifying the buffer / memory object permissions and remove the const keyword from the argument within the kernel. Most of the work for this program was modifying the kernel code to perform the panning operation. This required a bit of domain knowledge about how panning works. I also had a couple segfaults because when calling clSetKernelArg, I copy and pasted those lines and forgot to change the size of the argument being set and the index for that argument. However, print statements and the error checking code were quite useful for debugging when issues like that occured.

## Triumphs
A big triumph was getting to hear the output of the program. It was very satisfying to play with the panning values and observe the program working as expecting and varying what I heard in the output. In writing the panning kernel, I learned a bit about different methods for panning (equal power vs. linear), and chose to do equal power panning. I was also able to leverage opencl vectors, specifically the short2 datatype, since each frame in the audio data is a set of 2 short values, one for the left channel and another for the right channel. 
