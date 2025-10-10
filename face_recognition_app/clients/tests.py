from django.test import TestCase
from rest_framework.test import APIClient

from .models import Client, ClientAPIUsage


class ClientUsageLoggingMiddlewareTests(TestCase):
    def setUp(self):
        self.client_api = APIClient()
        self.client_obj = Client.objects.create(
            name="Middleware Client",
            description="",
            domain="https://example.com",
            contact_email="owner@example.com",
            contact_name="Owner",
        )
        self.client_api.credentials(HTTP_X_API_KEY=self.client_obj.api_key)

    def test_api_usage_logged_on_client_request(self):
        response = self.client_api.get("/api/core/info/")
        self.assertEqual(response.status_code, 200)

        usage_entries = ClientAPIUsage.objects.filter(client=self.client_obj)
        self.assertEqual(usage_entries.count(), 1)

        entry = usage_entries.first()
        self.assertEqual(entry.method, "GET")
        self.assertEqual(entry.status_code, 200)
        self.assertEqual(entry.endpoint, "analytics")
        self.assertTrue(entry.metadata.get("path", "").startswith("/api/core/info"))
