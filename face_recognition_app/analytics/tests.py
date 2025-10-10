from django.test import TestCase
from django.utils import timezone

from analytics.helpers import generate_daily_report
from auth_service.models import AuthenticationSession, FaceEnrollment, FaceRecognitionAttempt
from clients.models import Client, ClientUser


class AnalyticsHelpersTests(TestCase):
    def setUp(self):
        self.client_obj = Client.objects.create(
            name="Analytics Client",
            description="",
            domain="https://analytics.example.com",
            contact_email="ops@example.com",
            contact_name="Ops",
        )
        self.client_user = ClientUser.objects.create(
            client=self.client_obj,
            external_user_id="user-1",
            profile={},
        )

        self.enrollment_session = AuthenticationSession.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            session_type="enrollment",
            status="completed",
            metadata={"frames_processed": 12},
        )

        self.recognition_session = AuthenticationSession.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            session_type="recognition",
            status="completed",
            metadata={"frames_processed": 8},
        )

        self.enrollment = FaceEnrollment.objects.create(
            client=self.client_obj,
            client_user=self.client_user,
            enrollment_session=self.enrollment_session,
            status="active",
            embedding_vector="vector-data",
            face_quality_score=0.82,
            liveness_score=0.75,
            anti_spoofing_score=0.7,
            total_samples=1,
            sample_number=0,
            metadata={},
        )

        FaceRecognitionAttempt.objects.create(
            client=self.client_obj,
            session=self.recognition_session,
            result="success",
            matched_user=self.client_user,
            matched_enrollment=self.enrollment,
            similarity_score=0.88,
            confidence_score=0.9,
            face_quality_score=0.8,
            submitted_embedding="submitted-vector",
            processing_time_ms=120.0,
            metadata={},
        )

    def test_generate_daily_report_returns_expected_counts(self):
        report = generate_daily_report(self.client_obj, timezone.now().date())

        self.assertEqual(report["enrollments"]["completed"], 1)
        self.assertEqual(report["authentications"]["successful"], 1)
        self.assertEqual(report["total_frames_processed"], 20)
        self.assertGreaterEqual(report["quality_metrics"]["avg_enrollment_quality"], 0.8)
        self.assertGreater(report["success_rates"]["authentication_success_rate"], 0)
