"""
Firebase 연동을 위한 유틸리티 함수들
현재는 로컬 파일 시스템에 저장하고, 향후 Firebase 연동 구현 예정
"""

import json
import os
from datetime import datetime

# Firebase 연동 기능 주석 처리 - 향후 구현 예정
'''
def init_firebase_app(credentials_path=None):
    """
    Firebase 앱 초기화
    
    Args:
        credentials_path: Firebase 서비스 계정 키 파일 경로
        
    Returns:
        Firebase 앱 인스턴스
    """
    import firebase_admin
    from firebase_admin import credentials
    
    if credentials_path is None:
        # 환경 변수에서 서비스 계정 키 파일 경로 가져오기
        credentials_path = os.environ.get('FIREBASE_CREDENTIALS_PATH')
        
    cred = credentials.Certificate(credentials_path)
    firebase_app = firebase_admin.initialize_app(cred)
    
    return firebase_app
'''

def upload_json_to_firestore(collection_name, document_id, json_data):
    """
    JSON 데이터를 저장 (현재는 로컬 파일 시스템에 저장)
    향후 Firebase Firestore에 업로드 구현 예정
    
    Args:
        collection_name: 컬렉션 이름
        document_id: 문서 ID (None인 경우 자동 생성)
        json_data: 업로드할 JSON 데이터
        
    Returns:
        저장된 파일 경로
    """
    # Firebase 구현은 주석 처리
    '''
    from firebase_admin import firestore
    
    db = firestore.client()
    
    if document_id:
        doc_ref = db.collection(collection_name).document(document_id)
        doc_ref.set(json_data)
    else:
        doc_ref = db.collection(collection_name).add(json_data)
    
    return doc_ref
    '''
    
    # 로컬 파일에 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('output', 'firebase', collection_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if document_id:
        output_file = os.path.join(output_dir, f"{document_id}.json")
    else:
        output_file = os.path.join(output_dir, f"doc_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] JSON 데이터를 {output_file}에 저장했습니다. Firebase 연동 시 Firestore에 업로드 예정.")
    return output_file

def upload_image_to_storage(storage_path, image_path_or_bytes, content_type='image/jpeg'):
    """
    이미지를 저장 (현재는 로컬 파일 시스템에 저장)
    향후 Firebase Storage에 업로드 구현 예정
    
    Args:
        storage_path: Storage 경로 (예: 'images/user1/profile.jpg')
        image_path_or_bytes: 이미지 파일 경로 또는 바이트 데이터
        content_type: 이미지 MIME 타입
        
    Returns:
        저장된 파일 경로
    """
    # Firebase 구현은 주석 처리
    '''
    from firebase_admin import storage
    
    bucket = storage.bucket()
    blob = bucket.blob(storage_path)
    
    if isinstance(image_path_or_bytes, bytes):
        blob.upload_from_string(
            image_path_or_bytes,
            content_type=content_type
        )
    else:
        blob.upload_from_filename(
            image_path_or_bytes,
            content_type=content_type
        )
    
    # 파일을 공개 액세스 가능하게 설정
    blob.make_public()
    
    return blob.public_url
    '''
    
    # 로컬 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = os.path.dirname(storage_path)
    output_dir = os.path.join('output', 'firebase', 'storage', dir_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 파일 이름 처리
    file_name = os.path.basename(storage_path)
    local_path = os.path.join(output_dir, file_name)
    
    if isinstance(image_path_or_bytes, bytes):
        with open(local_path, 'wb') as f:
            f.write(image_path_or_bytes)
    else:
        import shutil
        
        # 파일이 존재하는지 확인
        if os.path.exists(image_path_or_bytes):
            shutil.copy2(image_path_or_bytes, local_path)
        else:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path_or_bytes}")
    
    print(f"[INFO] 이미지를 {local_path}에 저장했습니다. Firebase 연동 시 Storage에 업로드 예정.")
    return local_path
