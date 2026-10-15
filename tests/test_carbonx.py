"""Run with:  python -m unittest discover -s tests -v"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp = tempfile.mkdtemp()
os.environ["CARBONX_DB"] = os.path.join(_tmp, "cc.db")
shutil.copy2(ROOT / "cc.db", os.environ["CARBONX_DB"])

import benchmark  # noqa: E402
from app import app  # noqa: E402
from db import get_db  # noqa: E402

FIX = json.loads((ROOT / "tests" / "benchmark_fixtures.json").read_text())


def one(sql, *a):
    c = get_db()
    try:
        return c.execute(sql, a).fetchone()
    finally:
        c.close()


def snapshot_credits():
    c = get_db()
    try:
        return (
            [tuple(r) for r in c.execute("SELECT * FROM Carbon_Credit ORDER BY est_id")],
            [tuple(r) for r in c.execute("SELECT * FROM Credit_Wallet ORDER BY est_id")],
            c.execute("SELECT COUNT(*) FROM Credit_Ledger").fetchone()[0],
        )
    finally:
        c.close()


class BenchmarkModel(unittest.TestCase):
    def test_reproduces_research_predictions(self):
        for f in FIX:
            r = benchmark.predict(f["inputs"])
            self.assertAlmostEqual(r["expected_tco2e"], f["expected"], places=4)
            self.assertAlmostEqual(r["lower_tco2e"], f["lower"], places=3)
            self.assertAlmostEqual(r["upper_tco2e"], f["upper"], places=3)

    def test_interval_ordering_and_classification(self):
        r = benchmark.predict(FIX[0]["inputs"])
        self.assertLessEqual(r["lower_tco2e"], r["expected_tco2e"])
        self.assertLessEqual(r["expected_tco2e"], r["upper_tco2e"])
        lo, hi = r["lower_tco2e"], r["upper_tco2e"]
        self.assertEqual(benchmark.interval_comparison(hi + 1, lo, hi), benchmark.INTERVAL_ABOVE)
        self.assertEqual(benchmark.interval_comparison(lo - 1, lo, hi), benchmark.INTERVAL_BELOW)
        self.assertEqual(benchmark.interval_comparison((lo + hi) / 2, lo, hi), benchmark.INTERVAL_WITHIN)
        self.assertIsNone(benchmark.interval_comparison(None, lo, hi))

    def test_missing_optional_inputs_are_imputed_and_reported(self):
        x = dict(FIX[0]["inputs"])
        del x["occupancy"]
        r = benchmark.predict(x)
        self.assertIn("occupancy", r["imputed_fields"])

    def test_invalid_inputs_rejected(self):
        base = dict(FIX[0]["inputs"])
        for patch in ({"occupancy": 150}, {"occupancy": "nan"}, {"largest_property_use_type_1": 0},
                      {"largest_property_use_type_1": ""}, {"year_built": 3000},
                      {"primary_property_type_self": "Spaceport"}, {"borough": "Mysuru"},
                      {"number_of_buildings": "inf"}):
            with self.assertRaises(benchmark.BenchmarkError, msg=str(patch)):
                benchmark.predict({**base, **patch})
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark.predict(base, reported_tco2e=-5)


class Pages(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_every_page_loads(self):
        for url in ["/", "/emissions", "/report", "/trade", "/market", "/transactions",
                    "/add_establishment", "/add_activity", "/benchmark", "/benchmark/model"]:
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_internal_links_resolve(self):
        import re
        for url in ["/", "/report", "/emissions", "/benchmark"]:
            html = self.c.get(url).get_data(as_text=True)
            for href in set(re.findall(r'href="(/[^"#?]*)"', html)):
                if href.startswith("/static"):
                    continue
                self.assertEqual(self.c.get(href).status_code, 200, f"{url} -> {href}")

    def test_emissions_page_has_one_row_per_record(self):
        n = one("SELECT COUNT(*) FROM Emission_Record")[0]
        html = self.c.get("/emissions").get_data(as_text=True)
        self.assertEqual(html.count('class="em-row"'), n)

    def test_emission_equals_quantity_times_recorded_factor(self):
        bad = one("""SELECT COUNT(*) FROM Emission_Record er
                     JOIN Activity_Data ad USING(activity_id) JOIN Emission_Factor ef USING(factor_id)
                     WHERE ABS(er.emission_kg - ad.quantity*ef.factor_value) > 0.01""")[0]
        self.assertEqual(bad, 0)

    def test_report_marks_sectors_without_policy(self):
        html = self.c.get("/report").get_data(as_text=True)
        self.assertIn("No policy", html)
        self.assertIn("Simulated credits", html)


class BenchmarkRoutes(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()
        self.form = {k: v for k, v in FIX[1]["inputs"].items()}

    def test_benchmark_never_changes_credits(self):
        est = one("SELECT est_id FROM Establishment ORDER BY est_id LIMIT 1")[0]
        before = snapshot_credits()
        r = self.c.post("/benchmark", data={**self.form, "est_id": est, "period": "2026",
                                            "reported_tco2e": "5000"})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"No carbon credits were created", r.data)
        self.assertEqual(snapshot_credits(), before)
        row = one("SELECT * FROM Benchmark_Prediction WHERE est_id=? ORDER BY prediction_id DESC", est)
        self.assertIsNotNone(row)
        self.assertIsNotNone(one("SELECT 1 FROM Establishment_Profile WHERE est_id=?", est))

    def test_bad_input_returns_400_and_stores_nothing(self):
        n = one("SELECT COUNT(*) FROM Benchmark_Prediction")[0]
        r = self.c.post("/benchmark", data={**self.form, "occupancy": "500"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(one("SELECT COUNT(*) FROM Benchmark_Prediction")[0], n)

    def test_uses_recorded_emissions_when_reported_blank(self):
        row = one("""SELECT est_id, period, actual_kg FROM Establishment_Period_Emission
                     WHERE actual_kg > 0 LIMIT 1""")
        r = self.c.post("/benchmark", data={**self.form, "est_id": row["est_id"], "period": row["period"]})
        self.assertIn(b"recorded in CarbonX", r.data)


class Trading(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()
        conn = get_db()
        ids = [r[0] for r in conn.execute("SELECT est_id FROM Establishment ORDER BY est_id LIMIT 3")]
        self.s, self.b, self.b2 = ids
        # a clean funded seller with no listings
        conn.execute("UPDATE Credit_Listing SET status='Cancelled' WHERE seller_est_id=?", (self.s,))
        conn.execute("UPDATE Credit_Wallet SET available_credit=10, reserved_credit=0 WHERE est_id=?", (self.s,))
        conn.commit()
        conn.close()

    def wallet(self, est):
        r = one("SELECT available_credit, reserved_credit FROM Credit_Wallet WHERE est_id=?", est)
        return r["available_credit"], r["reserved_credit"]

    def listing(self, amount=4):
        self.c.post("/create-listing", data={"seller_est_id": self.s, "credit_amount": amount,
                                             "price_per_credit": 50})
        return one("SELECT listing_id FROM Credit_Listing WHERE seller_est_id=? AND status='Active' "
                   "ORDER BY listing_id DESC", self.s)[0]

    def test_nan_and_inf_rejected(self):
        for bad in ("nan", "inf", "-1", "0", "abc"):
            self.assertEqual(self.c.post("/create-listing", data={
                "seller_est_id": self.s, "credit_amount": bad, "price_per_credit": 5}).status_code, 400, bad)
            self.assertEqual(self.c.post("/trade", data={
                "seller_est_id": self.s, "buyer_est_id": self.b, "credit_amount": bad,
                "price_per_credit": 5}).status_code, 400, bad)

    def test_cannot_list_more_than_unreserved(self):
        self.listing(8)
        r = self.c.post("/create-listing", data={"seller_est_id": self.s, "credit_amount": 3,
                                                 "price_per_credit": 50})
        self.assertEqual(r.status_code, 400)

    def test_direct_trade_cannot_sell_reserved_credits(self):
        self.listing(8)  # 8 of 10 reserved
        r = self.c.post("/trade", data={"seller_est_id": self.s, "buyer_est_id": self.b,
                                        "credit_amount": 5, "price_per_credit": 10})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.wallet(self.s), (10, 8))

    def test_buy_moves_credits_to_buyer(self):
        lid = self.listing(4)
        a0 = self.wallet(self.b)[0]
        r = self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": self.b, "quantity": 1.5})
        self.assertEqual(r.status_code, 302)
        self.assertAlmostEqual(self.wallet(self.b)[0], a0 + 1.5)
        self.assertEqual(self.wallet(self.s), (8.5, 2.5))

    def test_listing_sells_out_exactly_once(self):
        lid = self.listing(4)
        self.assertEqual(self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": self.b,
                                                          "quantity": 4}).status_code, 302)
        self.assertEqual(one("SELECT status FROM Credit_Listing WHERE listing_id=?", lid)[0], "Sold")
        again = self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": self.b2, "quantity": 1})
        self.assertEqual(again.status_code, 400)
        self.assertEqual(self.wallet(self.s), (6, 0))

    def test_concurrent_buyers_cannot_oversell(self):
        lid = self.listing(4)
        codes = []

        def buy(buyer):
            codes.append(app.test_client().post("/buy-credit", data={
                "listing_id": lid, "buyer_est_id": buyer, "quantity": 3}).status_code)
        ts = [threading.Thread(target=buy, args=(x,)) for x in (self.b, self.b2)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(sorted(codes), [302, 400])
        a, r = self.wallet(self.s)
        self.assertEqual((a, r), (7, 1))

    def test_seller_cannot_buy_own_listing(self):
        lid = self.listing(2)
        self.assertEqual(self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": self.s,
                                                          "quantity": 1}).status_code, 400)

    def test_market_history_uses_volume_weighted_price(self):
        lid = self.listing(4)
        self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": self.b, "quantity": 2})
        row = one("SELECT avg_price FROM Market_History WHERE trade_date = DATE('now','localtime')")
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row[0], 50.0)


class DustListings(unittest.TestCase):
    """Regression: a listing with 8.7e-18 credits left showed '0.00 / Rs 0.00' with a dead Buy button."""

    def setUp(self):
        self.c = app.test_client()
        conn = get_db()
        self.est = conn.execute("SELECT est_id FROM Establishment ORDER BY est_id LIMIT 1").fetchone()[0]
        conn.execute("UPDATE Credit_Listing SET status='Cancelled' WHERE seller_est_id=?", (self.est,))
        conn.execute("UPDATE Credit_Wallet SET available_credit=5, reserved_credit=0 WHERE est_id=?", (self.est,))
        conn.commit()
        conn.close()

    def seed(self, remaining):
        conn = get_db()
        lid = conn.execute("""INSERT INTO Credit_Listing (seller_est_id, credit_amount, remaining_credit,
                              price_per_credit, status) VALUES (?, 2, ?, 50, 'Active')""",
                           (self.est, remaining)).lastrowid
        conn.execute("UPDATE Credit_Wallet SET reserved_credit=? WHERE est_id=?", (remaining, self.est))
        conn.commit()
        conn.close()
        return lid

    def test_float_residue_listing_is_closed_and_hidden(self):
        lid = self.seed(8.673617379884035e-18)
        self.assertNotIn(f'name="listing_id" value="{lid}"', self.c.get("/trade").get_data(as_text=True))
        import app as appmod
        appmod.initialize_database()
        row = one("SELECT status, remaining_credit FROM Credit_Listing WHERE listing_id=?", lid)
        self.assertEqual((row[0], row[1]), ("Sold", 0))
        self.assertEqual(one("SELECT reserved_credit FROM Credit_Wallet WHERE est_id=?", self.est)[0], 0)

    def test_sub_unit_remainder_is_cancelled_and_reservation_released(self):
        lid = self.seed(0.004)
        import app as appmod
        appmod.initialize_database()
        self.assertEqual(one("SELECT status FROM Credit_Listing WHERE listing_id=?", lid)[0], "Cancelled")
        self.assertEqual(one("SELECT reserved_credit FROM Credit_Wallet WHERE est_id=?", self.est)[0], 0)

    def test_normal_listing_untouched(self):
        lid = self.seed(0.5)
        import app as appmod
        appmod.initialize_database()
        self.assertEqual(one("SELECT status FROM Credit_Listing WHERE listing_id=?", lid)[0], "Active")
        self.assertEqual(one("SELECT reserved_credit FROM Credit_Wallet WHERE est_id=?", self.est)[0], 0.5)

    def test_buying_nearly_all_leaves_no_dust_listing(self):
        lid = self.seed(2)
        buyer = one("SELECT est_id FROM Establishment WHERE est_id<>? ORDER BY est_id LIMIT 1", self.est)[0]
        r = self.c.post("/buy-credit", data={"listing_id": lid, "buyer_est_id": buyer, "quantity": "1.999"})
        self.assertEqual(r.status_code, 302)          # rounds to 2.00, sells the lot
        self.assertEqual(one("SELECT status FROM Credit_Listing WHERE listing_id=?", lid)[0], "Sold")
        self.assertEqual(one("SELECT reserved_credit FROM Credit_Wallet WHERE est_id=?", self.est)[0], 0)

    def test_amount_below_one_unit_rejected(self):
        self.assertEqual(self.c.post("/create-listing", data={"seller_est_id": self.est,
                         "credit_amount": "0.004", "price_per_credit": 5}).status_code, 400)


class Establishments(unittest.TestCase):
    def setUp(self):
        self.c = app.test_client()

    def test_add_establishment_validates_and_creates_wallet(self):
        base = {"entity_name": "Test Plant", "entity_type": "Steel", "location": "Mysuru",
                "baseline": "5000", "baseline_year": "2025"}
        self.assertEqual(self.c.post("/add_establishment", data={**base, "entity_type": "Nope"}).status_code, 400)
        self.assertEqual(self.c.post("/add_establishment", data={**base, "baseline": "nan"}).status_code, 400)
        self.assertEqual(self.c.post("/add_establishment", data=base).status_code, 302)
        est = one("SELECT est_id FROM Establishment WHERE est_name='Test Plant'")[0]
        self.assertIsNotNone(one("SELECT 1 FROM Credit_Wallet WHERE est_id=?", est))
        self.assertIsNotNone(one("SELECT 1 FROM Carbon_Credit WHERE est_id=?", est))
        prov = one("""SELECT d.dataset_name FROM Baseline_Emission b JOIN Data_Provenance d USING(provenance_id)
                      WHERE b.est_id=?""", est)[0]
        self.assertIn("unverified", prov)

    def test_activity_without_factor_is_rolled_back(self):
        est = one("SELECT est_id FROM Establishment ORDER BY est_id LIMIT 1")[0]
        conn = get_db()
        sid = conn.execute("INSERT INTO Emission_Source (source_name, unit) VALUES ('NoFactor','kg')").lastrowid
        conn.commit()
        conn.close()
        n = one("SELECT COUNT(*) FROM Activity_Data")[0]
        r = self.c.post("/add_activity", data={"est_id": est, "source_id": sid, "quantity": 5, "period": "2026"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(one("SELECT COUNT(*) FROM Activity_Data")[0], n)

    def test_good_activity_creates_emission_record(self):
        est = one("SELECT est_id FROM Establishment ORDER BY est_id LIMIT 1")[0]
        n = one("SELECT COUNT(*) FROM Emission_Record")[0]
        r = self.c.post("/add_activity", data={"est_id": est, "source_id": 3, "quantity": 100, "period": "2026"})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(one("SELECT COUNT(*) FROM Emission_Record")[0], n + 1)


class PriceModel(unittest.TestCase):
    def test_forecast_declines_when_history_is_too_short(self):
        import ml_model
        r = ml_model.get_price_prediction()
        self.assertFalse(r["available"])

    def test_forecast_uses_latest_day_and_reports_naive_error(self):
        import ml_model
        conn = get_db()
        conn.execute("DELETE FROM Market_History")
        for i in range(30):
            conn.execute("""INSERT INTO Market_History (trade_date, avg_price, credits_traded, trade_value,
                            active_listings, credits_available) VALUES (?,?,?,?,?,?)""",
                         (f"2026-09-{i+1:02d}", 100 + i * 2.0, 5 + i % 3, 500 + i, 3, 10))
        conn.commit()
        conn.close()
        r = ml_model.get_price_prediction()
        self.assertTrue(r["available"])
        self.assertEqual(r["current_price"], 158.0)   # the true latest price (day 30)
        self.assertGreater(r["predicted_price"], 150)
        self.assertIn("naive_mae", r)


if __name__ == "__main__":
    unittest.main()
