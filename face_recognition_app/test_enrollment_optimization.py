#!/usr/bin/env python
"""
Test script for enrollment optimization components.
"""
import os
import sys

# Setup Django
os.environ['DJANGO_SETTINGS_MODULE'] = 'face_app.settings'

import django
django.setup()

def test_components():
    """Test all enrollment optimization components"""
    results = []
    
    # 1. Test OldPhotoEmbeddingExtractor
    try:
        from core.old_photo_extractor import OldPhotoEmbeddingExtractor, get_old_photo_extractor
        ext1 = get_old_photo_extractor()
        ext2 = get_old_photo_extractor()
        assert ext1 is ext2, "Singleton failed"
        results.append(("OldPhotoEmbeddingExtractor (singleton)", True, None))
    except Exception as e:
        results.append(("OldPhotoEmbeddingExtractor", False, str(e)))
    
    # 2. Test new Celery tasks
    try:
        from core.tasks import (
            extract_old_photo_embedding_task,
            bulk_extract_old_photo_embeddings, 
            preload_enrollment_user
        )
        results.append(("Celery tasks imported", True, None))
    except Exception as e:
        results.append(("Celery tasks", False, str(e)))
    
    # 3. Test signals
    try:
        import clients.signals
        results.append(("ClientUser signals", True, None))
    except Exception as e:
        results.append(("ClientUser signals", False, str(e)))
    
    # 4. Test consumers
    try:
        from auth_service.consumers import AuthProcessConsumer
        results.append(("AuthProcessConsumer", True, None))
    except Exception as e:
        results.append(("AuthProcessConsumer", False, str(e)))
    
    # 5. Test views preload endpoint
    try:
        from clients.views import ClientUserViewSet
        assert hasattr(ClientUserViewSet, 'preload'), "preload action missing"
        results.append(("ClientUserViewSet.preload", True, None))
    except Exception as e:
        results.append(("ClientUserViewSet.preload", False, str(e)))
    
    # 6. Test enrollment session view has preload trigger
    try:
        # Read the actual file source to check for preload
        import os
        views_path = os.path.join(os.path.dirname(__file__), 'auth_service', 'views.py')
        with open(views_path, 'r') as f:
            source = f.read()
        assert 'preload_enrollment_user' in source, "preload not in enrollment view"
        results.append(("Enrollment preload trigger", True, None))
    except Exception as e:
        results.append(("Enrollment preload trigger", False, str(e)))
    
    # Print results
    print("\n" + "=" * 60)
    print("ENROLLMENT OPTIMIZATION TEST RESULTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, success, error in results:
        if success:
            print(f"✓ {name}")
            passed += 1
        else:
            print(f"✗ {name}: {error}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = test_components()
    sys.exit(0 if success else 1)
