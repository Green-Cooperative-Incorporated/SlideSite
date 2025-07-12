import unittest
from app import app

class FlaskLoginTestCase(unittest.TestCase):
    def setUp(self):
        # Create a test client
        self.client = app.test_client()
        self.client.testing = True

    def test_successful_login_redirects_to_slidesite(self):
        response = self.client.post('/login', data=dict(
            username='admin',
            password='password123'
        ), follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Welcome to SlideSite', response.data)

    def test_failed_login_redirects_to_login(self):
        response = self.client.post('/login', data=dict(
            username='admin',
            password='wrongpassword'
        ), follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Invalid credentials', response.data)

if __name__ == '__main__':
    unittest.main()
