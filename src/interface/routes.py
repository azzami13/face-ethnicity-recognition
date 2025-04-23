from flask import Blueprint, request, jsonify
import numpy as np
import cv2
import base64

from src.face_detection.mtcnn_detector import MTCNNDetector
from src.face_similarity.facenet_embedder import FaceNetEmbedder
from src.ethnicity_detection.cnn_classifier import CNNEthnicityClassifier
from src.config import ETHNICITY_MAPPING_REVERSE, FACE_SIMILARITY_THRESHOLD

# Blueprint untuk rute API
api_routes = Blueprint('api', __name__)

# Inisialisasi model
detector = MTCNNDetector()
embedder = FaceNetEmbedder()
classifier = CNNEthnicityClassifier()

def decode_image(image_data):
    """
    Decode gambar dari base64 atau file upload
    """
    if isinstance(image_data, str):
        # Decode base64
        image_data = base64.b64decode(image_data.split(',')[1])
        img_array = np.frombuffer(image_data, np.uint8)
    else:
        # Dari file upload
        img_array = np.frombuffer(image_data.read(), np.uint8)
    
    # Decode gambar
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Konversi dari BGR ke RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

@api_routes.route('/detect_faces', methods=['POST'])
def detect_faces():
    """
    Endpoint untuk deteksi wajah
    """
    try:
        # Dapatkan gambar dari request
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'error': 'Tidak ada gambar yang diunggah',
                'status': 'failed'
            }), 400
        
        # Decode gambar
        image = decode_image(request.files['image'] if 'image' in request.files else request.json['image'])
        
        # Deteksi wajah
        faces = detector.extract_faces(image)
        
        # Siapkan response
        results = []
        for face_data in faces:
            results.append({
                'box': face_data['box'],
                'confidence': float(face_data['confidence']),
                'landmarks': face_data.get('landmarks', {})
            })
        
        return jsonify({
            'status': 'success',
            'faces': results,
            'total_faces': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_routes.route('/compare_faces', methods=['POST'])
def compare_faces():
    """
    Endpoint untuk perbandingan wajah
    """
    try:
        # Validasi input
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({
                'error': 'Diperlukan dua gambar untuk perbandingan',
                'status': 'failed'
            }), 400
        
        # Decode gambar
        image1 = decode_image(request.files['image1'])
        image2 = decode_image(request.files['image2'])
        
        # Deteksi wajah
        faces1 = detector.extract_faces(image1)
        faces2 = detector.extract_faces(image2)
        
        # Validasi deteksi wajah
        if not faces1 or not faces2:
            return jsonify({
                'status': 'failed',
                'error': 'Wajah tidak terdeteksi pada salah satu atau kedua gambar'
            }), 400
        
        # Ambil wajah pertama dari masing-masing gambar
        face1 = faces1[0]['face']
        face2 = faces2[0]['face']
        
        # Hitung similaritas
        similarity = embedder.compare_faces(face1, face2)
        is_same_person = similarity >= FACE_SIMILARITY_THRESHOLD
        
        return jsonify({
            'status': 'success',
            'similarity': float(similarity),
            'is_same_person': bool(is_same_person),
            'threshold': FACE_SIMILARITY_THRESHOLD
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@api_routes.route('/detect_ethnicity', methods=['POST'])
def detect_ethnicity():
    """
    Endpoint untuk deteksi etnis
    """
    try:
        # Validasi input
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({
                'error': 'Tidak ada gambar yang diunggah',
                'status': 'failed'
            }), 400
        
        # Decode gambar
        image = decode_image(request.files['image'] if 'image' in request.files else request.json['image'])
        
        # Deteksi wajah
        faces = detector.extract_faces(image)
        
        # Validasi deteksi wajah
        if not faces:
            return jsonify({
                'status': 'failed',
                'error': 'Tidak ada wajah yang terdeteksi'
            }), 400
        
        # Siapkan response
        results = []
        for face_data in faces:
            face = face_data['face']
            
            # Klasifikasi etnis
            class_idx, probabilities = classifier.predict(face)
            
            # Konversi hasil ke format yang dapat dibaca
            top_predictions = []
            sorted_indices = np.argsort(probabilities)[::-1]
            
            for idx in sorted_indices[:3]:  # Top 3 prediksi
                top_predictions.append({
                    'ethnicity': ETHNICITY_MAPPING_REVERSE.get(idx, f'Unknown {idx}'),
                    'probability': float(probabilities[idx])
                })
            
            results.append({
                'box': face_data['box'],
                'predicted_ethnicity': ETHNICITY_MAPPING_REVERSE.get(class_idx, f'Unknown {class_idx}'),
                'top_predictions': top_predictions
            })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'total_faces': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Error handler untuk route tidak ditemukan
@api_routes.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint tidak ditemukan'
    }), 404