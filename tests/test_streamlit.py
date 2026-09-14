"""Real Streamlit integration tests. Run in an environment with requirements installed."""
import unittest
from pathlib import Path
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest=None

@unittest.skipIf(AppTest is None,'Streamlit not installed in this runtime')
class StreamlitIntegrationTests(unittest.TestCase):
    def test_pages_and_individual_prediction(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        nav=app.sidebar.radio[0]
        for page in ['Prediksi Individu','Prediksi Batch','Kinerja Model']:
            nav.set_value(page).run()
            self.assertEqual(len(app.exception),0)
            nav=app.sidebar.radio[0]
        nav.set_value('Prediksi Individu').run()
        for button in app.button:
            if button.label=='Lihat hasil peninjauan':button.click().run();break
        self.assertEqual(len(app.exception),0)
        self.assertTrue(any(m.label=='Probabilitas Dropout' for m in app.metric))

if __name__=='__main__':unittest.main()
