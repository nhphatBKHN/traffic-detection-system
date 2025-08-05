# Traffic Congestion Detection at Urban Intersections

This project applies computer vision and deep learning techniques to detect and classify traffic congestion at urban intersections using surveillance video feeds.

## Features

- Vehicle detection using YOLOv3/YOLOv8.
- Traffic state classification based on motion and area occupancy.
- Real-time processing capabilities.
- ROI (Region of Interest) customization for different intersection layouts.

## Technologies

- Python 3
- OpenCV
- YOLO (You Only Look Once)
- NumPy
- Matplotlib
- TensorFlow / PyTorch (based on implementation)

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main program:
   ```bash
   python main.py
   ```

3. Configure your video input and ROI settings in `config.py` (if available).

## Dataset

Video footage of intersections collected locally. Due to privacy concerns, the dataset is not publicly shared. You can adapt the code to your own surveillance videos.

## Output

- Detected vehicles annotated in real-time.
- Traffic condition classification results.
- Visualization and logging of congestion trends.

## Authors

- Nguyen Huu Phat, Hanoi University of Science and Technology

## License

This project is for academic and research purposes.
