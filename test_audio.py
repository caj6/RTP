import numpy as np
import soundfile as sf
import io

# Creating a test audio signal
sample_rate = 16000
duration = 2.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), False)
test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Save to bytes
buffer = io.BytesIO()
sf.write(buffer, test_audio, sample_rate, format='WAV')
buffer.seek(0)

print("✅ Test audio created successfully")
print(f"Sample rate: {sample_rate} Hz")
print(f"Duration: {duration} seconds")
print(f"Audio size: {len(test_audio)} samples")

# Test loading
buffer.seek(0)
loaded_audio, loaded_rate = sf.read(buffer)
print(f"\n✅ Audio loaded back successfully")
print(f"Loaded sample rate: {loaded_rate} Hz")
print(f"Loaded samples: {len(loaded_audio)}")

# Save test file
sf.write("test_audio.wav", test_audio, sample_rate)
print("\n✅ Test file was saved as 'test_audio.wav'")
