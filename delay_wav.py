import wave
import sys
import struct

def decode_samples(frames, sample_width, total_samples):
    if sample_width == 3:
        # 24 bit: leggi 3 byte alla volta
        return [int.from_bytes(frames[i:i+3], byteorder='little', signed=True)
                for i in range(0, total_samples * 3, 3)]
    else:
        fmt = {1: 'b', 2: 'h', 4: 'i'}[sample_width]
        return list(struct.unpack(f'<{total_samples}{fmt}', frames))

def encode_samples(samples, sample_width):
    if sample_width == 3:
        result = bytearray()
        for s in samples:
            result += s.to_bytes(3, byteorder='little', signed=True)
        return bytes(result)
    else:
        fmt = {1: 'b', 2: 'h', 4: 'i'}[sample_width]
        return struct.pack(f'<{len(samples)}{fmt}', *samples)

def delay_wav_stereo(input_file, output_file, delay_samples):
    with wave.open(input_file, 'rb') as inp:
        params = inp.getparams()
        n_frames = inp.getnframes()
        frames = inp.readframes(n_frames)

    sample_width = params.sampwidth
    n_channels = params.nchannels
    total_samples = n_frames * n_channels

    samples = decode_samples(frames, sample_width, total_samples)

    # Se il file è già multicanale, usa solo il primo canale
    ch1 = samples[0::n_channels] if n_channels > 1 else samples

    # Canale 2: silenzio iniziale + ch1 con coda tagliata
    silence = [0] * delay_samples
    ch2 = (silence + ch1)[:len(ch1)]

    # Intercala i due canali: [L, R, L, R, ...]
    interleaved = []
    for s1, s2 in zip(ch1, ch2):
        interleaved.append(s1)
        interleaved.append(s2)

    out_frames = encode_samples(interleaved, sample_width)

    with wave.open(output_file, 'wb') as out:
        out.setnchannels(2)
        out.setsampwidth(sample_width)
        out.setframerate(params.framerate)
        out.writeframes(out_frames)

    print(f"Fatto! File stereo creato: ch1=originale, ch2=delay di {delay_samples} campioni")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python delay_wav.py input.wav output.wav num_campioni")
        sys.exit(1)

    delay_wav_stereo(sys.argv[1], sys.argv[2], int(sys.argv[3]))