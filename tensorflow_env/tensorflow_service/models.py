"""TensorFlow 모델 정의"""

import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Any


class MNISTModel:
    """간단한 MNIST 분류 모델"""
    
    def __init__(self, use_gpu: bool = True):
        """모델 초기화
        
        Args:
            use_gpu: GPU 사용 여부
        """
        # GPU 설정
        if use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # 메모리 증가를 허용
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"TensorFlow가 {len(gpus)}개의 GPU를 사용합니다.")
                except RuntimeError as e:
                    print(f"GPU 설정 오류: {e}")
            else:
                print("GPU를 찾을 수 없어 CPU를 사용합니다.")
        else:
            # CPU만 사용
            tf.config.set_visible_devices([], 'GPU')
            print("TensorFlow가 CPU를 사용합니다.")
        
        # 모델 생성
        self.model = self._create_model()
    
    def _create_model(self) -> tf.keras.Model:
        """모델 생성
        
        Returns:
            생성된 Keras 모델
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                                  input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """예측 수행
        
        Args:
            x: 입력 데이터 (numpy 배열)
            
        Returns:
            예측 결과와 메타데이터
        """
        # 입력 데이터 형태 조정
        if len(x.shape) == 2:  # (H, W) -> (H, W, 1)
            x = np.expand_dims(x, axis=-1)
        
        if len(x.shape) == 3 and x.shape[0] == 1:  # (1, H, W) -> (H, W, 1)
            x = np.transpose(x, (1, 2, 0))
        
        # 배치 차원 추가
        if len(x.shape) == 3:  # (H, W, C) -> (1, H, W, C)
            x = np.expand_dims(x, axis=0)
        
        # 예측 실행
        predictions = self.model.predict(x)
        
        # 클래스 인덱스 추출
        pred_classes = np.argmax(predictions, axis=1)
        
        # 메타데이터 생성
        metadata = {
            "confidence": np.max(predictions, axis=1).tolist(),
            "device": "GPU" if len(tf.config.list_physical_devices('GPU')) > 0 else "CPU",
            "framework": "tensorflow",
        }
        
        return pred_classes, metadata