import numpy as np
import soundfile as sf
import librosa
import io
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model

def extract_frame_features(file_path, frame_duration=1.0):
    video_clip = VideoFileClip(file_path)
    audio = video_clip.audio
    fps = audio.fps
    audio_samples = np.array(
        list(audio.iter_frames(fps=fps, dtype="float32"))
    ).flatten()
    buffer = io.BytesIO()
    sf.write(buffer, audio_samples, fps, format="wav")
    buffer.seek(0)
    x, sr = librosa.load(buffer, sr=None)

    # Split audio into frames of 'frame_duration' seconds
    frame_length = int(frame_duration * sr)
    frames = []
    timestamps = []

    for i in range(0, len(x), frame_length):
        if i + frame_length <= len(x):
            # Extract MFCCs for each frame and store the timestamp
            frame_mfcc = librosa.feature.mfcc(
                y=x[i : i + frame_length], sr=sr, n_mfcc=20
            )
            frames.append(frame_mfcc)
            timestamp = i / sr  # Convert index to seconds
            timestamps.append(timestamp)

    return frames, timestamps


def test_on_video(file_path, frame_duration=1.0):
    # Load the trained model
    model = load_model("model/TCN.keras")

    # Extract features and timestamps for each frame in the new video
    frames, timestamps = extract_frame_features(file_path, frame_duration)

    if frames is None or timestamps is None:
        print("No frames extracted.")
        return

    # Reshape frames for model input
    frames = np.array(frames)[..., np.newaxis]

    # Predict on each frame
    predictions = model.predict(frames)
    pred_labels = np.argmax(predictions, axis=1)

    # Store deepfake frames, their timestamps, and frame indices
    deepfake_frames = []
    deepfake_timestamps = []
    deepfake_indices = []

    # Identify deepfake frames
    for i, label in enumerate(pred_labels):
        if label == 1:  # If the label is FAKE
            deepfake_frames.append(frames[i])
            deepfake_timestamps.append(timestamps[i])
            deepfake_indices.append(i)

    if not deepfake_frames:
        print("No deepfake frames detected in the video.")
        return

    # Analyze deepfake frames
    print(f"Found {len(deepfake_frames)} deepfake frames:")
    for i, (timestamp, index) in enumerate(zip(deepfake_timestamps, deepfake_indices)):
        print(f"Frame {index + 1} at {timestamp:.2f}s: FAKE")


# Example usage
test_video_path = r"FAKE\aapnvogymq.mp4"  # Replace with your test video path
test_on_video(test_video_path)
