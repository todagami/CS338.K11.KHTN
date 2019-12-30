# CS338.K11.KHTN

- sinh viên sử dụng pyqt5 để tạo giao diện
- các yêu cầu cài đặt: 
+ python2/3
+ OpenCV
+ NumPy
+ scikit-learn
+ scikit-image
+ imutils
+ matplotlib
TensorFlow 2.0 (CPU or GPU)

##################################
build codd: python main.py (chạy giao diện)

code nằm trong cùng một thư mục có định dạng như bên dưới:

├── examples [25 entries]
├── gtsrb-german-traffic-sign
│   ├── Meta [43 entries]
│   ├── Test [12631 entries]
│   ├── Train [43 entries]
│   ├── meta-1 [43 entries]
│   ├── test-1 [12631 entries]
│   ├── train-1 [43 entries]
│   ├── Meta.csv
│   ├── Test.csv
│   └── Train.csv
├── output
│   ├── trafficsignnet.model
│   │   ├── assets
│   │   ├── variables
│   │   │   ├── variables.data-00000-of-00002
│   │   │   ├── variables.data-00001-of-00002
│   │   │   └── variables.index
│   │   └── saved_model.pb
│   └── plot.png
├── pyimagesearch
│   ├── __init__.py
│   └── trafficsignnet.py
├── train.py
├── signnames.csv
└── predict.py
 
gtsrb-german-traffic-sign/ : chưa bộ dataset
output/ : chưa kết quả output của model và training history plot được huấn luyên bởi train.py .
examples/ : chưa ảnh kết quả sau khi được predict 
pyimagesearch : một module: TrafficSignNet CNN.

