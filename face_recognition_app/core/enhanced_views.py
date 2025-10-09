"""
Enhanced API Views for Liveness and Obstacle Detection
"""
from rest_framework import generics, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.db.models import Prefetch
from django.utils import timezone
from datetime import timedelta
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse

from recognition.models import (
    AuthenticationAttempt, LivenessDetection, ObstacleDetection
)


@extend_schema(
    tags=['Liveness Detection'],
    summary='Get Liveness Detection History',
    description='Retrieve detailed liveness detection history for authenticated user',
    responses={
        200: OpenApiResponse(description='Liveness detection history'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class LivenessDetectionHistoryView(generics.ListAPIView):
    """Get liveness detection history"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return LivenessDetection.objects.filter(
            authentication_attempt__user=self.request.user
        ).select_related('authentication_attempt').order_by('-created_at')
    
    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()[:50]  # Limit to latest 50 records
        
        data = []
        for detection in queryset:
            attempt = detection.authentication_attempt
            data.append({
                'id': str(detection.id),
                'attempt_id': str(attempt.id),
                'session_id': attempt.session_id,
                'blinks_detected': detection.blinks_detected,
                'blink_quality_scores': detection.blink_quality_scores,
                'ear_history': detection.ear_history[-10:] if detection.ear_history else [],  # Last 10 EAR values
                'ear_baseline': detection.ear_baseline,
                'frames_processed': detection.frames_processed,
                'valid_frames': detection.valid_frames,
                'challenge_type': detection.challenge_type,
                'challenge_completed': detection.challenge_completed,
                'liveness_score': detection.liveness_score,
                'is_live': detection.is_live,
                'created_at': detection.created_at,
                'attempt_result': attempt.result,
                'similarity_score': attempt.similarity_score,
                'quality_score': attempt.quality_score,
                'debug_data': detection.debug_data
            })
        
        return Response({
            'count': len(data),
            'results': data
        })


@extend_schema(
    tags=['Obstacle Detection'],
    summary='Get Obstacle Detection History',
    description='Retrieve detailed obstacle detection history for authenticated user',
    responses={
        200: OpenApiResponse(description='Obstacle detection history'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
class ObstacleDetectionHistoryView(generics.ListAPIView):
    """Get obstacle detection history"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get_queryset(self):
        return ObstacleDetection.objects.filter(
            authentication_attempt__user=self.request.user
        ).select_related('authentication_attempt').order_by('-created_at')
    
    def list(self, request, *args, **kwargs):
        queryset = self.get_queryset()[:50]  # Limit to latest 50 records
        
        data = []
        for detection in queryset:
            attempt = detection.authentication_attempt
            obstacles_found = []
            
            if detection.glasses_detected:
                obstacles_found.append({
                    'type': 'glasses',
                    'confidence': detection.glasses_confidence
                })
            if detection.mask_detected:
                obstacles_found.append({
                    'type': 'mask',
                    'confidence': detection.mask_confidence
                })
            if detection.hat_detected:
                obstacles_found.append({
                    'type': 'hat',
                    'confidence': detection.hat_confidence
                })
            if detection.hand_covering:
                obstacles_found.append({
                    'type': 'hand_covering',
                    'confidence': detection.hand_confidence
                })
            
            data.append({
                'id': str(detection.id),
                'attempt_id': str(attempt.id),
                'session_id': attempt.session_id,
                'has_obstacles': detection.has_obstacles,
                'obstacle_score': detection.obstacle_score,
                'obstacles_found': obstacles_found,
                'glasses_detected': detection.glasses_detected,
                'glasses_confidence': detection.glasses_confidence,
                'mask_detected': detection.mask_detected,
                'mask_confidence': detection.mask_confidence,
                'hat_detected': detection.hat_detected,
                'hat_confidence': detection.hat_confidence,
                'hand_covering': detection.hand_covering,
                'hand_confidence': detection.hand_confidence,
                'detection_details': detection.detection_details,
                'created_at': detection.created_at,
                'attempt_result': attempt.result,
                'similarity_score': attempt.similarity_score,
                'quality_score': attempt.quality_score
            })
        
        return Response({
            'count': len(data),
            'results': data
        })


@extend_schema(
    tags=['Analytics'],
    summary='Get Detection Analytics',
    description='Get analytics data for liveness and obstacle detection',
    parameters=[
        OpenApiParameter(
            name='days',
            type=int,
            location=OpenApiParameter.QUERY,
            description='Number of days to include in analytics (default: 30)',
            required=False
        )
    ],
    responses={
        200: OpenApiResponse(description='Detection analytics'),
        401: OpenApiResponse(description='Authentication required'),
    }
)
@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def detection_analytics_view(request):
    """Get analytics for liveness and obstacle detection"""
    days = int(request.GET.get('days', 30))
    start_date = timezone.now() - timedelta(days=days)
    
    # Get authentication attempts for the user
    attempts = AuthenticationAttempt.objects.filter(
        user=request.user,
        created_at__gte=start_date
    ).prefetch_related(
        Prefetch('liveness_detection'),
        Prefetch('obstacle_detection')
    )
    
    # Analytics data
    total_attempts = attempts.count()
    successful_attempts = attempts.filter(result='success').count()
    failed_liveness = 0
    failed_obstacles = 0
    
    # Liveness analytics
    liveness_stats = {
        'total_sessions_with_liveness': 0,
        'avg_blinks_per_session': 0,
        'avg_liveness_score': 0,
        'liveness_success_rate': 0,
        'common_challenges': []
    }
    
    # Obstacle analytics  
    obstacle_stats = {
        'sessions_with_obstacles': 0,
        'obstacle_types': {
            'glasses': {'count': 0, 'avg_confidence': 0},
            'mask': {'count': 0, 'avg_confidence': 0},
            'hat': {'count': 0, 'avg_confidence': 0},
            'hand_covering': {'count': 0, 'avg_confidence': 0}
        },
        'most_common_obstacle': None
    }
    
    total_blinks = 0
    total_liveness_score = 0
    liveness_count = 0
    
    for attempt in attempts:
        # Liveness data
        try:
            liveness = attempt.liveness_detection
            if liveness:
                liveness_stats['total_sessions_with_liveness'] += 1
                total_blinks += liveness.blinks_detected
                total_liveness_score += liveness.liveness_score
                liveness_count += 1
                
                if not liveness.is_live and attempt.result.startswith('failed'):
                    failed_liveness += 1
        except LivenessDetection.DoesNotExist:
            pass
        
        # Obstacle data
        try:
            obstacles = attempt.obstacle_detection
            if obstacles and obstacles.has_obstacles:
                obstacle_stats['sessions_with_obstacles'] += 1
                
                if obstacles.glasses_detected:
                    obstacle_stats['obstacle_types']['glasses']['count'] += 1
                    obstacle_stats['obstacle_types']['glasses']['avg_confidence'] += obstacles.glasses_confidence
                
                if obstacles.mask_detected:
                    obstacle_stats['obstacle_types']['mask']['count'] += 1
                    obstacle_stats['obstacle_types']['mask']['avg_confidence'] += obstacles.mask_confidence
                
                if obstacles.hat_detected:
                    obstacle_stats['obstacle_types']['hat']['count'] += 1
                    obstacle_stats['obstacle_types']['hat']['avg_confidence'] += obstacles.hat_confidence
                
                if obstacles.hand_covering:
                    obstacle_stats['obstacle_types']['hand_covering']['count'] += 1
                    obstacle_stats['obstacle_types']['hand_covering']['avg_confidence'] += obstacles.hand_confidence
                
                if attempt.result == 'failed_obstacles':
                    failed_obstacles += 1
        except ObstacleDetection.DoesNotExist:
            pass
    
    # Calculate averages
    if liveness_count > 0:
        liveness_stats['avg_blinks_per_session'] = round(total_blinks / liveness_count, 1)
        liveness_stats['avg_liveness_score'] = round(total_liveness_score / liveness_count, 2)
        liveness_stats['liveness_success_rate'] = round(
            ((liveness_count - failed_liveness) / liveness_count) * 100, 1
        )
    
    # Calculate obstacle confidence averages and find most common
    most_common_count = 0
    most_common_type = None
    
    for obstacle_type, data in obstacle_stats['obstacle_types'].items():
        if data['count'] > 0:
            data['avg_confidence'] = round(data['avg_confidence'] / data['count'], 2)
            if data['count'] > most_common_count:
                most_common_count = data['count']
                most_common_type = obstacle_type
    
    obstacle_stats['most_common_obstacle'] = most_common_type
    
    return Response({
        'period': {
            'days': days,
            'start_date': start_date,
            'end_date': timezone.now()
        },
        'overall': {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': round((successful_attempts / total_attempts) * 100, 1) if total_attempts > 0 else 0,
            'failed_liveness': failed_liveness,
            'failed_obstacles': failed_obstacles
        },
        'liveness': liveness_stats,
        'obstacles': obstacle_stats
    })