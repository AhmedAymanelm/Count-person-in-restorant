# Person Counter for Restaurant 🍽️👥

A sophisticated person counting system designed for restaurants using YOLO11x and DeepSORT with Re-ID capabilities.

## 🚀 Features

- **Advanced Detection**: YOLO11x for accurate person detection
- **Smart Tracking**: DeepSORT with Re-ID for robust person tracking
- **Entry/Exit Counting**: Automatic counting of people entering and exiting
- **Trajectory Visualization**: Visual tracking of person movement paths
- **Performance Optimized**: Skip frames and memory management
- **Error Handling**: Comprehensive error handling and validation
- **CSV Logging**: Detailed tracking data export
- **Configurable**: Easy parameter adjustment

## 📋 Requirements

```bash
pip install ultralytics
pip install deep-sort-realtime
pip install opencv-python
pip install torch
pip install torchreid or pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
pip install  tensorboard
pip install gdown
pip install Cython

```

## 🎯 Usage

### Basic Usage
```bash
python main.py
```

### Advanced Usage
```bash
python main.py --video "your_video.mp4" --model "yolo11x.pt" --skip-frames 2
```

### Command Line Arguments
- `--video`: Path to input video file (default: "WhatsApp Video 2025-09-01 at 16.56.14.mp4")
- `--model`: Path to YOLO model file (default: "yolo11x.pt")
- `--skip-frames`: Number of frames to skip for performance (default: 2)

## ⚙️ Configuration

You can modify the following parameters in the `Config` class:

```python
class Config:
    # Performance settings
    SKIP_FRAMES = 2
    MAX_TRAJECTORY_POINTS = 50
    
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5
    
    # Tracking settings
    MAX_AGE = 200
    N_INIT = 3
    MAX_IOU_DISTANCE = 0.6
    
    # Line coordinates (x1, y1, x2, y2)
    LIMIT_ENTER = [700, 300, 900, 700]
    LIMIT_EXIT = [600, 300, 700, 600]
    
    # Ignore zone (x1, y1, x2, y2)
    IGNORE_ZONE = (0, 0, 200, 500)
```

## 📁 Output Files

- `person_output.mp4`: Processed video with tracking visualization
- `tracking_log.csv`: Detailed tracking data with timestamps

## 🔧 Technical Details

### Architecture
1. **Video Input**: Validates and loads video file
2. **Person Detection**: YOLO11x detects people in each frame
3. **Tracking**: DeepSORT maintains person identity across frames
4. **Line Crossing**: Detects when people cross entry/exit lines
5. **Visualization**: Draws bounding boxes, trajectories, and counters
6. **Output**: Saves processed video and CSV log

### Performance Optimizations
- Frame skipping for faster processing
- Limited trajectory points to save memory
- Efficient bounding box calculations
- Optimized video codec selection

### Error Handling
- File validation
- Model loading error handling
- Video writer initialization
- CSV logging error recovery

## 📊 CSV Output Format

The CSV file contains the following columns:
- `frame`: Frame number
- `track_id`: Unique person identifier
- `x1, y1, x2, y2`: Bounding box coordinates
- `cx, cy`: Center point coordinates
- `status`: Person status (Active/Inactive)

## 🎨 Visualization Features

- **Bounding Boxes**: Semi-transparent colored boxes around detected people
- **Person IDs**: Unique identifier labels
- **Trajectories**: Movement path visualization
- **Entry/Exit Lines**: Green (enter) and red (exit) lines
- **Counters**: Real-time enter/exit/active person counts
- **Progress Bar**: Processing progress indicator

## 🚨 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model file not found: yolo11x.pt
   Solution: Ensure the model file is in the project directory
   ```

2. **Video file not found**
   ```
   Error: Video file not found: your_video.mp4
   Solution: Check the video file path
   ```

3. **Performance issues**
   ```
   Solution: Increase SKIP_FRAMES value in Config class
   ```

4. **Memory issues**
   ```
   Solution: Decrease MAX_TRAJECTORY_POINTS in Config class
   ```

## 📈 Performance Metrics

- **Processing Speed**: ~15-30 FPS (depending on hardware)
- **Memory Usage**: Optimized with trajectory point limits
- **Accuracy**: High accuracy with YOLO11x + DeepSORT
- **Scalability**: Configurable for different video resolutions

- File Gender and age [here] : https://drive.google.com/drive/folders/11GqrrHvK_isKcMwpEpNcOldj3VkweApk

## 🔮 Future Enhancements

- [ ] Real-time webcam support
- [ ] Multi-camera synchronization
- [ ] Advanced analytics dashboard
- [ ] Crowd density estimation
- [ ] Time-based analytics
- [ ] Alert system for capacity limits

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support and questions, please open an issue on the project repository.

---

**Made with ❤️ for better restaurant management** 
