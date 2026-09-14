"""Regression tests for audited failures and deployment contracts."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from student_success.config import ROOT,DATA_PATH,ARTIFACT_DIR,REPORT_DIR
from student_success.schema import FEATURES,FIELDS,defaults,validate_features,ValidationError,parse_csv
from student_success.inference import load_bundle,predict_frame,action_label,reference_warnings
from student_success.train import EXPECTED_DATA_SHA256
from student_success.inference import sha256_file

class DataContractTests(unittest.TestCase):
    def setUp(self):self.frame=pd.DataFrame([defaults()])
    def test_synthetic_presets_are_coherent(self):
        for name in ['Contoh umum','Perlu dukungan akademik','Akademik kuat']:
            x=validate_features(pd.DataFrame([defaults(name)]))
            self.assertTrue((x.Curricular_units_1st_sem_approved<=x.Curricular_units_1st_sem_enrolled).all())
    def test_invalid_numbers_are_rejected(self):
        for c,v in [('Age_at_enrollment',-5),('Course',999),('Admission_grade',999),('Admission_grade',np.inf),('Admission_grade',np.nan),('Application_order',1.5),('Curricular_units_1st_sem_grade','abc')]:
            with self.subTest(column=c,value=v):
                x=self.frame.copy();x[c]=v
                with self.assertRaises(ValidationError):validate_features(x)
    def test_academic_relation(self):
        x=self.frame.copy();x.Curricular_units_1st_sem_approved=7
        with self.assertRaises(ValidationError):validate_features(x)
    def test_without_evaluations_relation(self):
        x=self.frame.copy();x.Curricular_units_1st_sem_without_evaluations=8
        with self.assertRaises(ValidationError):validate_features(x)
    def test_empty_missing_duplicate(self):
        for x in [self.frame.iloc[:0],self.frame.drop(columns='Course'),pd.concat([self.frame,self.frame[['Course']]],axis=1)]:
            with self.assertRaises(ValidationError):validate_features(x)
    def test_csv_formats_and_duplicate_header(self):
        for sep in [',',';']:
            actual=validate_features(parse_csv(self.frame.to_csv(index=False,sep=sep).encode('utf-8-sig')))
            pd.testing.assert_frame_equal(actual,validate_features(self.frame))
        with self.assertRaises(ValidationError):parse_csv(b'Course,Course\n1,1\n')
    def test_extra_columns_never_enter_model(self):
        x=self.frame.copy();x['Status']='Dropout';x['Status_encoded']=0;x['Curricular_units_2nd_sem_grade']=20
        self.assertEqual(list(validate_features(x)),FEATURES)
    def test_historical_data_matches_contract(self):
        data=pd.read_csv(DATA_PATH,sep=';')
        self.assertEqual(sha256_file(DATA_PATH),EXPECTED_DATA_SHA256)
        self.assertEqual(len(validate_features(data)),4424)
    def test_policy_boundaries(self):
        self.assertEqual(action_label(.189999,.19),'Pemantauan rutin')
        self.assertEqual(action_label(.19,.19),'Perlu peninjauan')
        self.assertEqual(action_label(.40,.19),'Perlu peninjauan')

class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model,cls.manifest=load_bundle()
    def test_individual_equals_batch_and_feature_reorder(self):
        x=pd.read_csv(ROOT/'examples'/'students_template.csv')
        batch=predict_frame(x,self.model,self.manifest)
        permuted=predict_frame(x[x.columns[::-1]],self.model,self.manifest)
        pd.testing.assert_frame_equal(batch,permuted)
        for i in range(len(x)):
            one=predict_frame(x.iloc[[i]],self.model,self.manifest)
            np.testing.assert_allclose(one[['prob_dropout','prob_enrolled','prob_graduate']].iloc[0],batch[['prob_dropout','prob_enrolled','prob_graduate']].iloc[i])
            self.assertEqual(one.action.iloc[0],batch.action.iloc[i])
    def test_artifact_reproduces_recorded_holdout_probabilities(self):
        saved=pd.read_csv(REPORT_DIR/'historical_holdout_predictions.csv')
        data=pd.read_csv(DATA_PATH,sep=';').iloc[saved.source_row.to_numpy()-1]
        result=predict_frame(data,self.model,self.manifest)
        cols=['prob_dropout','prob_enrolled','prob_graduate']
        np.testing.assert_allclose(result[cols],saved[cols],rtol=1e-7,atol=1e-8)
    def test_metadata_and_split_integrity(self):
        splits=pd.read_csv(REPORT_DIR/'split_assignments.csv')
        self.assertEqual(splits.source_row.nunique(),4424)
        self.assertEqual(len(splits),4424)
        self.assertEqual(splits.split.value_counts().to_dict(),{'model_development':2654,'policy_validation':885,'historical_holdout':885})
        self.assertFalse(any('2nd_sem' in c for c in self.manifest['features']))
        self.assertNotIn('Debtor',self.manifest['features'])
        self.assertNotIn('Tuition_fees_up_to_date',self.manifest['features'])
    def test_threshold_derived_from_policy_validation(self):
        table=pd.read_csv(REPORT_DIR/'threshold_analysis.csv')
        row=table.sort_values(['f2','threshold'],ascending=False).iloc[0]
        self.assertAlmostEqual(row.threshold,self.manifest['threshold'])
    def test_checksum_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'manifest.json').write_text((ARTIFACT_DIR/'manifest.json').read_text())
            (p/'model.joblib').write_bytes(b'not the model')
            with self.assertRaisesRegex(ValueError,'Checksum'):load_bundle(p)
    def test_domain_and_reference_ranges_are_distinct(self):
        x=pd.DataFrame([defaults()]);x.Age_at_enrollment=90
        validate_features(x)
        self.assertTrue('Age_at_enrollment' in reference_warnings(x,self.manifest).column.tolist())

if __name__=='__main__':unittest.main()
