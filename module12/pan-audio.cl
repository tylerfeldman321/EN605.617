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
    __global short2* audio,
    const int num_frames,
    const float pan)
{
    size_t id = get_global_id(0);

    if (id >= num_frames)
        return;

	// Using equal power panning
	float channel_1_gain = sin(pan * PI * 0.5);
    float channel_2_gain = cos(pan * PI * 0.5);

    short2 frame = audio[id];
    float left = frame.x * channel_1_gain;
    float right = frame.y * channel_2_gain;

    audio[id] = (short2)(left, right);
}