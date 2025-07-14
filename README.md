# CNN Leaf Disease Detection

## Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to automatically detect and classify potato leaf diseases from images. The model can identify three classes: **Healthy leaves**, **Early Blight**, and **Late Blight** diseases in potato plants.

## Main Purpose

The primary objective is to develop an intelligent agricultural system that can:
- Automatically diagnose potato leaf diseases from digital images
- Assist farmers in early disease detection to prevent crop loss
- Provide accurate classification with confidence scores
- Support precision agriculture and sustainable farming practices

## Dataset Structure

The project uses a comprehensive potato leaf dataset organized into three categories:
- **Potato___healthy**: Healthy potato leaves
- **Potato___Early_blight**: Leaves affected by Early Blight disease
- **Potato___Late_blight**: Leaves affected by Late Blight disease

## Model Development Steps

### 1. Data Preprocessing
- **Image Loading**: Images loaded using `tf.keras.preprocessing.image_dataset_from_directory`
- **Image Resizing**: All images standardized to 256x256 pixels
- **Normalization**: Pixel values scaled to [0,1] range using `Rescaling(1.0/255)`
- **Data Augmentation**: Applied horizontal/vertical flips, random rotation (0.2), and random zoom (0.1)

### 2. Dataset Splitting
- **Training Set**: 80% of the data (54 batches)
- **Validation Set**: 10% of the data (6 batches)
- **Test Set**: 10% of the data (remaining batches)
- **Batch Size**: 32 images per batch
- **Optimization**: Applied caching, shuffling, and prefetching for efficient training

### 3. CNN Architecture
The model consists of the following layers:

```
Sequential Model:
├── Input Layer (32, 256, 256, 3)
├── Resize and Rescaling
├── Data Augmentation
├── Conv2D (32 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Conv2D (64 filters, 3x3, ReLU)
├── MaxPooling2D (2x2)
├── Flatten
├── Dense (64 units, ReLU)
└── Dense (3 units, Softmax) - Output Layer
```

### 4. Model Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 50
- **Validation**: Real-time validation during training

### 5. Model Evaluation and Testing
- Performance evaluation on test dataset
- Individual image prediction with confidence scores
- Visual comparison of actual vs predicted labels
- Training/validation accuracy and loss visualization

## Key Features

- **Multi-class Classification**: Identifies 3 different categories of potato leaf conditions
- **Data Augmentation**: Improves model generalization and robustness
- **Real-time Prediction**: Provides instant disease classification
- **Confidence Scoring**: Shows prediction confidence percentage
- **Visualization**: Displays training progress and prediction results

## Model Output

The trained model provides:
- **Classification Results**: Predicts one of three classes (Healthy, Early Blight, Late Blight)
- **Confidence Scores**: Percentage confidence for each prediction
- **Performance Metrics**: Training and validation accuracy/loss curves
- **Saved Model**: Trained model saved as `1.keras` in the models directory

## FastAPI Implementation

### API Overview
The project includes a FastAPI-based REST API that allows users to upload potato leaf images and receive real-time disease classification results. The API provides a simple interface for integrating the trained CNN model into web applications, mobile apps, or other systems.

### API Endpoints

#### 1. Health Check Endpoint
- **URL**: `GET /ping`
- **Description**: Simple health check to verify the API is running
- **Response**: 
  ```json
  {
    "message": "Hello, World!"
  }
  ```

#### 2. Image Upload and Prediction Endpoint
- **URL**: `POST /uploadfile/`
- **Description**: Upload a potato leaf image for disease classification
- **Request**: Multipart form data with an image file
- **Response**:
  ```json
  {
    "class": "Potato___Early_blight",
    "confidence": 0.95,
    "filename": "potato_leaf.jpg"
  }
  ```

### API Features

- **Real-time Prediction**: Instant classification of uploaded images
- **Image Preprocessing**: Automatic resizing to 256×256 pixels and normalization
- **Confidence Scoring**: Returns prediction confidence as a percentage
- **Error Handling**: Graceful error handling with descriptive error messages
- **File Support**: Accepts common image formats (JPG, PNG, etc.)

### Running the API

1. **Navigate to the API directory**:
   ```bash
   cd API
   ```

2. **Start the FastAPI server**:
   ```bash
   python main.py
   ```

3. **Access the API**:
   - Server runs on: `http://localhost:8000`
   - Interactive API docs: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

### API Usage Examples

#### Using cURL
```bash
# Health check
curl -X GET "http://localhost:8000/ping"

# Upload image for prediction
curl -X POST "http://localhost:8000/uploadfile/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@potato_leaf_image.jpg"
```

#### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/ping")
print(response.json())

# Upload image for prediction
with open("potato_leaf_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/uploadfile/", files=files)
    print(response.json())
```

#### Using JavaScript/Fetch
```javascript
// Upload image for prediction
const formData = new FormData();
formData.append('file', imageFile);

fetch('http://localhost:8000/uploadfile/', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

### API Architecture

The FastAPI implementation includes:

1. **Image Processing Pipeline**:
   - File upload handling
   - Image format validation
   - Automatic resizing to 256×256 pixels
   - Pixel normalization (0-1 range)

2. **Model Integration**:
   - Local model loading from `models/1.keras`
   - Batch prediction processing
   - Class name mapping

3. **Response Processing**:
   - Confidence score calculation
   - JSON response formatting
   - Error handling and logging

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhinav61/CNN_Leave_Detection.git
   cd CNN_Leave_Detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook Agriculture.ipynb
   ```

### GPU Support (Optional)
For faster training with GPU support, ensure you have CUDA installed and uncomment the tensorflow-gpu line in `requirements.txt`.

## Technologies Used

- **TensorFlow & Keras**: Deep learning framework
- **FastAPI**: Modern, fast web framework for building APIs
- **Python**: Programming language
- **Matplotlib**: Data visualization
- **NumPy**: Numerical computations
- **OpenCV/PIL**: Image processing
- **Jupyter Notebook**: Development environment
- **Scikit-learn**: Machine learning utilities
- **Uvicorn**: ASGI server for FastAPI

## Applications

This model and API can be integrated into:
- **Web Applications**: Upload images through web interface
- **Mobile Applications**: Smartphone apps for farmers in the field
- **Agricultural Monitoring Systems**: Automated crop health monitoring
- **Precision Farming Tools**: IoT devices with camera integration
- **Educational Platforms**: Agricultural studies and training programs
- **Early Warning Systems**: Automated crop disease detection alerts
- **Third-party Software**: Integration via REST API endpoints

## Future Enhancements

- Extension to other crop diseases
- Mobile app development
- Real-time camera integration
- Treatment recommendation system
- Multi-language support for global farmers 
