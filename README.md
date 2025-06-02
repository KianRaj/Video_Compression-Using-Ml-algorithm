# Video Compression Using Unsupervised Learning

This project implements a basic video compression pipeline leveraging **unsupervised learning** techniques and motion estimation. It uses **KMeans clustering** for color quantization on image blocks combined with **block-based motion estimation** (diamond search algorithm) to compress video frames efficiently.

---

## Overview

Video compression reduces the size of video data by exploiting spatial and temporal redundancies. This project applies the following techniques:

- **Spatial Compression**: Each video frame is divided into fixed-size blocks (e.g., 16x16 pixels). For the first frame, the average color of each block is calculated, and KMeans clustering groups similar blocks to reduce color space complexity.
  
- **Temporal Compression**: For subsequent frames, block-based motion estimation (diamond search algorithm) finds matching blocks in the previous frame, encoding motion vectors instead of full pixel data.
  
- **Residual Frames**: The difference between the actual frame and motion-predicted frame is calculated and stored as residuals to maintain visual quality.

Together, these approaches reduce the overall video data size while maintaining reconstruction quality.

---

## Key Components

### 1. Image Blocking
The frame is split into blocks of fixed size (`block_size`), which are processed individually for color quantization and motion estimation.


### 2. Color Quantization with KMeans
The average color of each block in the first frame is computed, then clustered using KMeans into a small set of representative colors (num_clusters).

average_colors_first = calculate_average_color(blocks_first)
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(average_colors_first)

### 3. Motion Estimation (Diamond Search)
For every block in the current frame, the diamond search algorithm locates the best matching block in the previous frame within a search window.

### 4. Residual Frame Calculation
Residual frames store the difference between the actual frame and the predicted frame obtained from motion compensation.


### 6. Encoding & Decoding Pipeline
Encoded data includes cluster assignments for color blocks, motion vectors, block size, and residual frames.

Decoding reconstructs frames by applying motion vectors to previous frames and adding residuals.

Periodic refresh frames (every refresh_rate frames) ensure error does not accumulate.

### 7. Performance Metric: Signal-to-Noise Ratio (SNR)
After reconstruction, the average Peak Signal-to-Noise Ratio (PSNR) between original and decompressed frames is computed to evaluate quality.

# How to Run
Set the path to your input video file:
video_path = "Give Your Datset path in code.py"
Adjust parameters if needed:

block_size: Tuple for block width and height (e.g., (16, 16))

num_clusters: Number of color clusters for KMeans (e.g., 5)

seconds_to_process: Number of seconds of video to compress

frame_rate: Video frame rate (e.g., 24)

Run the compression script. It will:

Read video frames

Compress frames using clustering and motion vectors

Save compressed data to compressed_data.pkl

Load compressed data and decompress frames to reconstruct the video

Save the reconstructed video as 'reconstructed_video2.avi'

View the average PSNR for quality assessment

Dependencies
Python 3.x

OpenCV (cv2)

NumPy

scikit-learn (sklearn)

scikit-image (skimage)

pickle (standard library)

Install dependencies via pip if needed:

bash
Copy
pip install opencv-python numpy scikit-learn scikit-image
Project Structure
image_to_blocks(): Splits frames into blocks

calculate_average_color(): Computes average block colors

diamond_search(): Performs block motion estimation

calculate_motion_vectors(): Gets motion vectors for all blocks

calculate_residual_frame(): Computes difference between frames

encode_video_data(): Prepares data for storage

apply_motion_vectors_to_frame(): Reconstructs predicted frames using motion vectors

Main loop: Compress and decompress video frames with periodic refreshes

PSNR calculation for quality measurement

## Limitations & Future Work
Current compression is simple and not optimized for production use.

Uses fixed block size and number of clusters.

Does not implement entropy coding or more advanced prediction schemes.

Future improvements:

Adaptive block sizes

More sophisticated clustering or vector quantization

Integration with codecs like H.264 or HEVC

GPU acceleration for faster processing





```python
def image_to_blocks(img, block_size):
    img_height, img_width, _ = img.shape
    blocks = []
    for y in range(0, img_height, block_size[1]):
        for x in range(0, img_width, block_size[0]):
            block = img[y:y+block_size[1], x:x+block_size[0]]
            blocks.append(block)
    return blocks

def calculate_average_color(blocks):
    average_colors = []
    for block in blocks:
        average_color = np.mean(block, axis=(0, 1))
        average_colors.append(average_color)
    return np.array(average_colors)

def reconstruct_image_from_clusters(cluster_assignments, cluster_centers, block_size, frame_shape):
    reconstructed_img = np.zeros(frame_shape, dtype=np.uint8)
    block_index = 0
    for y in range(0, frame_shape[0], block_size[1]):
        for x in range(0, frame_shape[1], block_size[0]):
            if y + block_size[1] <= frame_shape[0] and x + block_size[0] <= frame_shape[1]:
                reconstructed_img[y:y + block_size[1], x:x + block_size[0]] = cluster_centers[cluster_assignments[block_index]]
            block_index += 1
    return reconstructed_img

def calculate_motion_vectors(first_frame, next_frame, block_size):
    blocks_first = image_to_blocks(first_frame, block_size)
    motion_vectors = []
    for idx, block in enumerate(blocks_first):
        block_y = (idx // (first_frame.shape[1] // block_size[0])) * block_size[1]
        block_x = (idx % (first_frame.shape[1] // block_size[0])) * block_size[0]
        mv = diamond_search(block, next_frame, block_x, block_y, 5)
        motion_vectors.append(mv)
    return motion_vectors

def calculate_residual_frame(actual_frame, predicted_frame):
    """Calculate the residual (difference) frame."""
    residual = cv2.subtract(actual_frame, predicted_frame)
    return residual

def apply_residual_frame(predicted_frame, residual):
    """Apply the residual frame to the predicted frame."""
    reconstructed_frame = cv2.add(predicted_frame, residual)
    return reconstructed_frame

def encode_video_data(cluster_assignments, motion_vectors, block_size, frame_shape):
    motion_model = np.array(motion_vectors).flatten()
    return {
        'clusters': cluster_assignments.tolist(),
        'motion_model': motion_model.tolist(),
        'block_size': block_size,
        'frame_shape': frame_shape
    }

def apply_motion_vectors_to_frame(frame, motion_vectors, block_size):
    new_frame = np.zeros_like(frame)
    num_blocks_y, num_blocks_x = frame.shape[0] // block_size[1], frame.shape[1] // block_size[0]

    .........
    .......

    return new_frame

def diamond_search(block, ref_frame, block_pos_x, block_pos_y, max_search_range):
    block_height, block_width, _ = block.shape
    ref_height, ref_width, _ = ref_frame.shape

    small_diamond = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]
    large_diamond = [(-2, 0), (0, -2), (2, 0), (0, 2)] + small_diamond

    def get_sad(center_x, center_y):
        y1, y2 = center_y, center_y + block_height
        x1, x2 = center_x, center_x + block_width
        if 0 <= x1 < ref_width and 0 <= y1 < ref_height and x2 <= ref_width and y2 <= ref_height:
            candidate_block = ref_frame[y1:y2, x1:x2]
            return np.sum(np.abs(block.astype(np.int32) - candidate_block.astype(np.int32)))
        else:
      ..........
..........................


"""----------------------------------------Split here for ipynb--------------------------------------"""


#Add location of video in video_path
video_path = "/content/drive/MyDrive/VIDEO_COMPRESSION/VIDEOS/AlitaBattleAngel.mkv"
block_size = (16, 16)
num_clusters = 5
seconds_to_process = 60
frame_rate = 24
cap = cv2.VideoCapture(video_path)
.....
......


"""----------------------------------------Split here for ipynb--------------------------------------"""


# First pass to process video and store compressed data
for _ in range(seconds_to_process * frame_rate - 1):
    ret, next_frame = cap.read()
    if not ret:
        break

    count += 1
    print(f"Processing frame {count}")
    original_frames.append(next_frame)
    motion_vectors = calculate_motion_vectors(first_frame, next_frame, block_size)
    encoded_data = encode_video_data(cluster_assignments_first, motion_vectors, block_size, first_frame.shape)
    compressed_data.append(encoded_data)

    first_frame = next_frame

total_frames = len(original_frames) - 1
compressed_data = []
...
...
...

# Save compressed_data to a file
compressed_data_path = 'compressed_data.pkl'

with open(compressed_data_path, 'wb') as file:
    pickle.dump(compressed_data, file)

print(f"Compressed data saved to {compressed_data_path}")

compressed_data_path = 'compressed_data.pkl'


"""----------------------------------------Split here for ipynb--------------------------------------"""


# Load compressed_data from the file
with open(compressed_data_path, 'rb') as file:
    compressed_data = pickle.load(file)

print("Compressed data successfully loaded.")
..........................................
...........................................
import cv2

output_video_path = 'reconstructed_video2.avi'
codec = cv2.VideoWriter_fourcc(*'XVID')
output_frame_rate = frame_rate

print(f"Total frames to write: {len(decompressed_frames)}")
if decompressed_frames:
    print(f"Frame size: {decompressed_frames[0].shape}")

output_size = (decompressed_frames[0].shape[1], decompressed_frames[0].shape[0])
out = cv2.VideoWriter(output_video_path, codec, output_frame_rate, output_size)

for idx, frame in enumerate(decompressed_frames):
    print(f"Writing frame {idx + 1}/{len(decompressed_frames)}")
    out.write(frame)

out.release()

print(f"Video reconstructed and saved to {output_video_path}")


"""----------------------------------------Split here for ipynb--------------------------------------"""


snr_values = [compare_psnr(orig, decomp) for orig, decomp in zip(original_frames, decompressed_frames)]
average_snr = sum(snr_values) / len(snr_values)
cap.release()

print(f"Average SNR: {average_snr} dB")


"""----------------------------------------------- END ----------------------------------------------



