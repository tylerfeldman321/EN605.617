// pan-audio.cl
//
//    Pans audio

#define PI 3.14159265358979323846f

//  pan_audio_2channel
//		pan - float from 0.0 (full left) to 1.0 (full right)
//
//
//	Assumes global work size is larger than number of frames
__kernel void pan_audio_2channel(
    __global short* interleaved_in,
    const int num_frames,
    const float pan)
{
    size_t id = get_global_id(0);

    if (id >= num_frames)
        return;

	short channel_1_sample = interleaved_in[2 * id];
	short channel_2_sample = interleaved_in[2 * id + 1];

	// Using equal power panning
	float channel_1_gain = sin(pan * PI * 0.5);
    float channel_2_gain = cos(pan * PI * 0.5);

	channel_1_sample = (short)(channel_1_sample * channel_1_gain);
    channel_2_sample = (short)(channel_2_sample * channel_2_gain);

    interleaved_in[id * 2] = channel_1_sample;
    interleaved_in[id * 2 + 1] = channel_2_sample;
}