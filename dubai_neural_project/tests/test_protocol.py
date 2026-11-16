"""Invariants that protect the scientific comparison, not implementation mirrors."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from core import load_data,time_split,features,preprocessor,corrupt,PriceModel
ROOT=Path(__file__).resolve().parents[1]

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=load_data(ROOT/"data/off_plan.csv")
        cls.tr,cls.va,cls.te=time_split(cls.raw)

    def test_holdouts_have_no_rows_or_dates_in_common(self):
        self.assertEqual((len(self.tr),len(self.va),len(self.te)),(6738,2278,2984))
        self.assertLess(self.tr.date_listed.max(),self.va.date_listed.min())
        self.assertLess(self.va.date_listed.max(),self.te.date_listed.min())
        self.assertFalse(set(self.tr.id)&set(self.va.id))
        self.assertFalse(set(self.tr.id)&set(self.te.id))

    def test_target_and_ids_cannot_change_feature_matrix(self):
        d=self.te.head(10).copy();x=features(d)
        for c in ["price_usd","price_per_m2_usd","price_per_sqft_usd","id","project_id","project_name"]:
            d[c]="forbidden input"
        pd.testing.assert_frame_equal(x,features(d))

    def test_preprocessing_does_not_refit_on_future(self):
        p=preprocessor().fit(features(self.tr));before=p.named_transformers_["num"].named_steps["scale"].mean_.copy()
        future=self.te.head(4).copy();future["area_m2"]=1e10;future["view"]="NEVER_OBSERVED"
        transformed=p.transform(features(future))
        np.testing.assert_array_equal(before,p.named_transformers_["num"].named_steps["scale"].mean_)
        self.assertTrue(np.isfinite(transformed).all())

    def test_corruption_is_exact_nested_and_leaves_truth(self):
        d=self.te.reset_index(drop=True)
        a=corrupt(d,"area",.05,11);b=corrupt(d,"area",.2,11)
        ia=set(np.flatnonzero(a.area_m2!=d.area_m2));ib=set(np.flatnonzero(b.area_m2!=d.area_m2))
        self.assertEqual(len(ia),round(.05*len(d)));self.assertTrue(ia<=ib)
        np.testing.assert_array_equal(a.price_usd,d.price_usd)
        np.testing.assert_allclose(a.area_m2,a.area_sqft*.09290304)
        missing=corrupt(d,"missing",.1,11)
        for c in ["area_m2","metro_distance_min"]:
            self.assertEqual(missing[c].isna().sum(),round(.1*len(d)))
        c=corrupt(d,"view",.1,11,{"view":set(self.tr.view)})
        self.assertEqual((c.view!=d.view).sum(),round(.1*len(d)))

    def test_unit_price_prediction_uses_observed_area(self):
        m=PriceModel("Median","unit",{}).fit(self.tr,self.va)
        a=self.te.head(4).copy();b=a.copy();b["area_m2"]*=2
        np.testing.assert_allclose(m.predict(b),2*m.predict(a))
        b["area_m2"]=np.nan
        self.assertTrue(np.isfinite(m.predict(b)).all())

if __name__=="__main__":unittest.main(verbosity=2)
