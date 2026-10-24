import json
import math
import os
import sqlite3
from contextlib import closing

from flask import Flask, abort, jsonify, redirect, render_template, request
from chatbot import (
    answer_question,
    ChatbotConfigurationError,
    ChatbotProviderError,
)
import benchmark
from db import get_db, write_transaction
from ml_model import get_price_prediction

app = Flask(__name__)

EPS = 1e-9
CREDIT_UNIT = 0.01  # smallest tradable amount; credits are shown and traded to 2 decimals


def close_dust_listings(conn):
    """Close listings whose remainder is below one tradable unit.

    Old purchases subtracted floats and could leave remaining_credit = 8.7e-18 while the
    listing stayed 'Active', so it showed "0.00 credits / Rs 0.00" with a Buy button that
    could never work. Such a listing is closed and any reservation it still held is released.
    """
    rows = conn.execute("""SELECT listing_id, seller_est_id, remaining_credit FROM Credit_Listing
                           WHERE status = 'Active' AND remaining_credit < ?""",
                        (CREDIT_UNIT - EPS,)).fetchall()
    for r in rows:
        noise = r["remaining_credit"] <= 1e-6          # float residue of a fully sold listing
        conn.execute("UPDATE Credit_Listing SET status = ?, remaining_credit = 0 WHERE listing_id = ?",
                     ("Sold" if noise else "Cancelled", r["listing_id"]))
        conn.execute("""UPDATE Credit_Wallet
                         SET reserved_credit = ROUND(MAX(reserved_credit - ?, 0), 6),
                             updated_at = CURRENT_TIMESTAMP
                         WHERE est_id = ?""", (r["remaining_credit"], r["seller_est_id"]))
    # reservations must equal the sum of live listings; remove leftover float residue
    conn.execute("""UPDATE Credit_Wallet
        SET reserved_credit = ROUND(COALESCE((SELECT SUM(remaining_credit) FROM Credit_Listing l
                                              WHERE l.seller_est_id = Credit_Wallet.est_id
                                                AND l.status = 'Active'), 0), 6)
        WHERE ABS(reserved_credit - ROUND(COALESCE((SELECT SUM(remaining_credit) FROM Credit_Listing l
                                              WHERE l.seller_est_id = Credit_Wallet.est_id
                                                AND l.status = 'Active'), 0), 6)) < 0.000001""")


def ensure_wallet_system(conn):
    columns = [row["name"] for row in conn.execute("PRAGMA table_info(Credit_Wallet)").fetchall()]
    if "generated_credit" not in columns:
        conn.execute("ALTER TABLE Credit_Wallet ADD COLUMN generated_credit REAL NOT NULL DEFAULT 0")

    # every establishment has exactly one credit row and one wallet row
    conn.execute("""
        INSERT OR IGNORE INTO Carbon_Credit (est_id, credit, status)
        SELECT est_id, 0, 'Neutral' FROM Establishment""")
    conn.execute("""
        INSERT OR IGNORE INTO Credit_Wallet (est_id, available_credit, reserved_credit, generated_credit)
        SELECT est_id, 0, 0, 0 FROM Establishment""")

    conn.execute("""
        UPDATE Credit_Wallet
        SET generated_credit = COALESCE((
            SELECT MAX(COALESCE(cc.credit, 0), 0) FROM Carbon_Credit cc
            WHERE cc.est_id = Credit_Wallet.est_id), 0)""")
    conn.execute("""
        UPDATE Credit_Wallet
        SET available_credit = MAX(
                reserved_credit,
                generated_credit + COALESCE((
                    SELECT SUM(cl.change_amount) FROM Credit_Ledger cl
                    WHERE cl.est_id = Credit_Wallet.est_id
                      AND cl.transaction_type IN ('Credit Purchase', 'Credit Sale')), 0)),
            updated_at = CURRENT_TIMESTAMP""")

    conn.execute("DROP TRIGGER IF EXISTS log_credit_change")
    conn.execute("DROP TRIGGER IF EXISTS sync_wallet_from_credit")
    conn.execute("""
        CREATE TRIGGER sync_wallet_from_credit
        AFTER UPDATE OF credit ON Carbon_Credit
        FOR EACH ROW
        WHEN ROUND(COALESCE(NEW.credit, 0) - COALESCE(OLD.credit, 0), 10) != 0
        BEGIN
            INSERT INTO Credit_Ledger (est_id, change_amount, transaction_type, description)
            VALUES (NEW.est_id,
                    ROUND(MAX(COALESCE(NEW.credit, 0), 0) - MAX(COALESCE(OLD.credit, 0), 0), 10),
                    'Emission Adjustment',
                    'Generated carbon-credit balance changed after emission recalculation');
            UPDATE Credit_Wallet
            SET generated_credit = MAX(COALESCE(NEW.credit, 0), 0),
                available_credit = MAX(reserved_credit,
                    available_credit + (MAX(COALESCE(NEW.credit, 0), 0)
                                        - MAX(COALESCE(OLD.credit, 0), 0))),
                updated_at = CURRENT_TIMESTAMP
            WHERE est_id = NEW.est_id;
        END""")
    conn.commit()


def initialize_database():
    with closing(get_db()) as conn:
        ensure_wallet_system(conn)
        close_dust_listings(conn)
        conn.commit()


initialize_database()

def to_float(value, name, *, positive=False, non_negative=False):
    """Parse a form number; reject NaN/inf, which float() happily accepts."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number.")
    if positive and v <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    if non_negative and v < 0:
        raise ValueError(f"{name} cannot be negative.")
    return v


def to_credits(value, name):
    """Credit quantity: finite, rounded to 2 decimals, at least one tradable unit (0.01)."""
    v = round(to_float(value, name, positive=True), 2)
    if v < CREDIT_UNIT:
        raise ValueError(f"{name} must be at least {CREDIT_UNIT}.")
    return v


def to_int(value, name):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} is invalid.")


def error(message, status=400):
    return message, status


def update_market_history(conn):
    """Refresh today's Market_History row (local date, consistently)."""
    market = conn.execute("""
        SELECT DATE(created_at, 'localtime') AS trade_date,
               SUM(total_amount) / NULLIF(SUM(credit_amount), 0) AS avg_price,
               SUM(credit_amount) AS credits_traded,
               SUM(total_amount) AS trade_value
        FROM Credit_Transaction
        WHERE status = 'Completed'
          AND DATE(created_at, 'localtime') = DATE('now', 'localtime')""").fetchone()
    if market["avg_price"] is None:
        return
    active_listings = conn.execute(
        "SELECT COUNT(*) AS c FROM Credit_Listing WHERE status = 'Active'").fetchone()["c"]
    credits_available = conn.execute(
        "SELECT COALESCE(SUM(remaining_credit), 0) AS c FROM Credit_Listing WHERE status = 'Active'"
    ).fetchone()["c"]
    conn.execute("""
        INSERT INTO Market_History
            (trade_date, avg_price, credits_traded, trade_value, active_listings, credits_available)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_date) DO UPDATE SET
            avg_price = excluded.avg_price, credits_traded = excluded.credits_traded,
            trade_value = excluded.trade_value, active_listings = excluded.active_listings,
            credits_available = excluded.credits_available""",
        (market["trade_date"], market["avg_price"], market["credits_traded"],
         market["trade_value"], active_listings, credits_available))


def get_market_trend(conn):
    rows = conn.execute("""
        SELECT trade_date, avg_price, credits_traded, trade_value
        FROM Market_History ORDER BY trade_date ASC""").fetchall()
    if len(rows) < 2:
        return {"direction": "STABLE", "price_change": 0, "total_volume": 0,
                "description": "Not enough historical data to determine market direction."}
    first_price, latest_price = rows[0]["avg_price"], rows[-1]["avg_price"]
    price_change = ((latest_price - first_price) / first_price * 100) if first_price else 0
    if price_change > 5:
        direction, description = "BULLISH", "Carbon credit prices are trending upward."
    elif price_change < -5:
        direction, description = "BEARISH", "Carbon credit prices are trending downward."
    else:
        direction, description = "STABLE", "Carbon credit prices are relatively stable."
    return {"direction": direction, "price_change": round(price_change, 2),
            "description": description,
            "total_volume": sum(r["credits_traded"] for r in rows)}


@app.route("/")
def dashboard():
    with closing(get_db()) as conn:
        total_est = conn.execute("SELECT COUNT(*) FROM Establishment").fetchone()[0]
        total_act = conn.execute("SELECT COUNT(*) FROM Activity_Data").fetchone()[0]
        total_em = conn.execute("SELECT COUNT(*) FROM Emission_Record").fetchone()[0]
    return render_template("dashboard.html", total_est=total_est,
                           total_act=total_act, total_em=total_em)


@app.route("/add_establishment", methods=["GET", "POST"])
def add_establishment():
    with closing(get_db()) as conn:
        sectors = conn.execute(
            "SELECT DISTINCT sector FROM Reduction_Policy ORDER BY sector").fetchall()
        if request.method == "POST":
            try:
                name = (request.form.get("entity_name") or "").strip()
                est_type = (request.form.get("entity_type") or "").strip()
                location = (request.form.get("location") or "").strip()
                if not name or not location:
                    raise ValueError("Name and location are required.")
                if est_type not in {s["sector"] for s in sectors}:
                    raise ValueError("Choose one of the listed establishment types.")
                baseline = to_float(request.form.get("baseline") or 9000, "Baseline",
                                    non_negative=True)
                year = to_int(request.form.get("baseline_year") or 2024, "Baseline year")
                if not 1900 <= year <= 2100:
                    raise ValueError("Baseline year must be between 1900 and 2100.")
            except ValueError as e:
                return error(str(e))

            with write_transaction() as w:
                prov = w.execute("SELECT provenance_id FROM Data_Provenance WHERE dataset_name = ?",
                                 ("User-declared baseline (unverified)",)).fetchone()
                prov_id = prov["provenance_id"] if prov else w.execute("""
                    INSERT INTO Data_Provenance (dataset_name, organization, methodology, is_synthetic)
                    VALUES ('User-declared baseline (unverified)', 'CarbonX users',
                            'Entered through the add-establishment form; no evidence attached', 0)
                    """).lastrowid
                est_id = w.execute(
                    "INSERT INTO Establishment (est_name, est_type, location) VALUES (?,?,?)",
                    (name, est_type, location)).lastrowid
                w.execute("""INSERT INTO Baseline_Emission
                             (est_id, baseline_emission_kg, baseline_year, provenance_id)
                             VALUES (?,?,?,?)""", (est_id, baseline, year, prov_id))
                w.execute("INSERT INTO Carbon_Credit (est_id, credit, status) VALUES (?, 0, 'Neutral')",
                          (est_id,))
                w.execute("""INSERT INTO Credit_Wallet
                             (est_id, available_credit, reserved_credit, generated_credit)
                             VALUES (?, 0, 0, 0)""", (est_id,))
            return redirect("/")
    return render_template("add_entity.html", sectors=sectors)


@app.route("/add_activity", methods=["GET", "POST"])
def add_activity():
    with closing(get_db()) as conn:
        if request.method == "POST":
            try:
                est_id = to_int(request.form.get("est_id"), "Establishment")
                source_id = to_int(request.form.get("source_id"), "Source")
                quantity = to_float(request.form.get("quantity"), "Quantity", positive=True)
                period = (request.form.get("period") or "").strip()
                if not (period.isdigit() and len(period) == 4 and 1900 <= int(period) <= 2100):
                    raise ValueError("Period must be a four-digit year.")
                if not conn.execute("SELECT 1 FROM Establishment WHERE est_id=?", (est_id,)).fetchone():
                    raise ValueError("Unknown establishment.")
                if not conn.execute("SELECT 1 FROM Emission_Source WHERE source_id=?", (source_id,)).fetchone():
                    raise ValueError("Unknown emission source.")
            except ValueError as e:
                return error(str(e))

            try:
                with write_transaction() as w:
                    activity_id = w.execute(
                        "INSERT INTO Activity_Data (est_id, source_id, quantity, period) VALUES (?,?,?,?)",
                        (est_id, source_id, quantity, period)).lastrowid
                    if not w.execute("SELECT 1 FROM Emission_Record WHERE activity_id=?",
                                     (activity_id,)).fetchone():
                        raise ValueError("No active emission factor exists for that source.")
            except ValueError as e:
                return error(str(e))
            return redirect("/emissions")

        establishments = conn.execute(
            "SELECT est_id, est_name FROM Establishment ORDER BY est_name").fetchall()
        sources = conn.execute(
            "SELECT source_id, source_name FROM Emission_Source ORDER BY source_name").fetchall()
    return render_template("add_activity.html", establishments=establishments, sources=sources)


@app.route("/emissions")
def emissions():
    with closing(get_db()) as conn:
        rows = conn.execute("""
            SELECT e.est_name, s.source_name, s.unit, ad.quantity, ef.factor_value, ef.effective_year,
                   er.emission_kg
            FROM Emission_Record er
            JOIN Activity_Data ad ON er.activity_id = ad.activity_id
            JOIN Establishment e ON ad.est_id = e.est_id
            JOIN Emission_Source s ON ad.source_id = s.source_id
            LEFT JOIN Emission_Factor ef ON er.factor_id = ef.factor_id
            ORDER BY e.est_name, ad.period""").fetchall()
    clean_rows = [{
        "est_name": r["est_name"], "source_name": r["source_name"], "unit": r["unit"],
        "quantity": r["quantity"], "factor_value": r["factor_value"],
        "emission_kg": round(r["emission_kg"] or 0, 2),
    } for r in rows]
    return render_template("emissions.html", rows=clean_rows)


@app.route("/report")
def report():
    with closing(get_db()) as conn:
        rows = conn.execute("""
            SELECT v.*, EXISTS(SELECT 1 FROM Reduction_Policy rp WHERE rp.sector = v.sector)
                        AS has_policy
            FROM Carbon_Report_View v""").fetchall()
    data = []
    for r in rows:
        if r["activity_count"] == 0:
            credit, actual, status = 0, "No activity", "No Activity"
        else:
            credit = r["carbon_credit"] if r["carbon_credit"] is not None else 0
            actual = r["actual_emission"]
            status = "Surplus" if credit > 0 else "Deficit" if credit < 0 else "Neutral"
        data.append({
            "name": r["est_name"], "sector": r["sector"], "reduction": r["reduction_percent"] or 0,
            "baseline": r["baseline_emission_kg"], "allowed": r["allowed_limit"],
            "actual": actual, "credit": round(credit, 2), "status": status,
            "has_policy": bool(r["has_policy"]),
        })
    return render_template("report.html", rows=data)

BENCH_FORM_FIELDS = list(benchmark.NUMERIC_FIELDS) + ["primary_property_type_self", "borough"]


def _benchmark_context(conn):
    cfg = benchmark.load_config()
    return {
        "property_types": cfg["property_types"], "boroughs": cfg["boroughs"],
        "establishments": conn.execute(
            "SELECT est_id, est_name FROM Establishment ORDER BY est_name").fetchall(),
        "cfg": cfg,
    }


@app.route("/benchmark", methods=["GET", "POST"])
def benchmark_page():
    with closing(get_db()) as conn:
        ctx = _benchmark_context(conn)
        ctx.update(result=None, form={}, message=None, est=None, actual_tco2e=None)
        est_id_arg = request.args.get("est_id")
        if request.method == "GET" and est_id_arg:
            prof = conn.execute("SELECT * FROM Establishment_Profile WHERE est_id=?",
                                (est_id_arg,)).fetchone()
            if prof:
                ctx["form"] = {
                    "est_id": est_id_arg, "primary_property_type_self": prof["property_type"],
                    "borough": prof["borough"],
                    "largest_property_use_type_1": prof["largest_use_area_sqft"],
                    "_2nd_largest_property_use_1": prof["second_use_area_sqft"],
                    "_3rd_largest_property_use_1": prof["third_use_area_sqft"],
                    "year_built": prof["year_built"], "number_of_buildings": prof["number_of_buildings"],
                    "occupancy": prof["occupancy_pct"]}
            else:
                ctx["form"] = {"est_id": est_id_arg}

        if request.method == "POST":
            form = {k: request.form.get(k, "") for k in BENCH_FORM_FIELDS}
            form.update(est_id=request.form.get("est_id", ""), period=request.form.get("period", ""),
                        reported_tco2e=request.form.get("reported_tco2e", ""))
            ctx["form"] = form
            try:
                est_id = None
                if form["est_id"]:
                    est_id = to_int(form["est_id"], "Establishment")
                    if not conn.execute("SELECT 1 FROM Establishment WHERE est_id=?", (est_id,)).fetchone():
                        raise ValueError("Unknown establishment.")
                reported = None
                period = form["period"].strip() or None
                if form["reported_tco2e"].strip():
                    reported = to_float(form["reported_tco2e"], "Reported emissions", non_negative=True)
                elif est_id and period:
                    # use the emissions CarbonX has actually recorded for that year
                    row = conn.execute(
                        "SELECT actual_kg FROM Establishment_Period_Emission WHERE est_id=? AND period=?",
                        (est_id, period)).fetchone()
                    if row and row["actual_kg"] is not None:
                        reported = row["actual_kg"] / 1000.0
                        ctx["actual_tco2e"] = reported
                result = benchmark.predict(form, reported)
            except (ValueError, benchmark.BenchmarkError) as e:
                ctx["message"] = str(e)
                return render_template("benchmark.html", **ctx), 400

            ctx["result"] = result
            if est_id:
                inp = result["inputs"]
                model_id = conn.execute(
                    "SELECT model_id FROM Benchmark_Model WHERE version=?",
                    (result["model_version"],)).fetchone()["model_id"]
                with write_transaction() as w:
                    w.execute("""
                        INSERT INTO Establishment_Profile
                            (est_id, property_type, borough, largest_use_area_sqft, second_use_area_sqft,
                             third_use_area_sqft, year_built, number_of_buildings, occupancy_pct, updated_at)
                        VALUES (?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                        ON CONFLICT(est_id) DO UPDATE SET
                            property_type=excluded.property_type, borough=excluded.borough,
                            largest_use_area_sqft=excluded.largest_use_area_sqft,
                            second_use_area_sqft=excluded.second_use_area_sqft,
                            third_use_area_sqft=excluded.third_use_area_sqft,
                            year_built=excluded.year_built,
                            number_of_buildings=excluded.number_of_buildings,
                            occupancy_pct=excluded.occupancy_pct, updated_at=CURRENT_TIMESTAMP""",
                        (est_id, inp["primary_property_type_self"], inp["borough"],
                         inp["largest_property_use_type_1"], inp["_2nd_largest_property_use_1"],
                         inp["_3rd_largest_property_use_1"], int(inp["year_built"]),
                         int(inp["number_of_buildings"]), inp["occupancy"]))
                    w.execute("""
                        INSERT INTO Benchmark_Prediction
                            (est_id, model_id, period, expected_tco2e, lower_tco2e, upper_tco2e,
                             reported_tco2e, comparison, review_action, inputs_json, imputed_fields)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (est_id, model_id, period, result["expected_tco2e"], result["lower_tco2e"],
                         result["upper_tco2e"], result["reported_tco2e"], result["comparison"],
                         result["review_action"], json.dumps(inp), ",".join(result["imputed_fields"])))
        history = []
        if ctx["form"].get("est_id"):
            history = conn.execute("""
                SELECT created_at, period, expected_tco2e, lower_tco2e, upper_tco2e,
                       reported_tco2e, comparison
                FROM Benchmark_Prediction WHERE est_id=? ORDER BY prediction_id DESC LIMIT 10""",
                (ctx["form"]["est_id"],)).fetchall()
        ctx["history"] = history
        return render_template("benchmark.html", **ctx)


@app.route("/benchmark/model")
def benchmark_model():
    with closing(get_db()) as conn:
        model = conn.execute("SELECT * FROM Benchmark_Model ORDER BY model_id DESC LIMIT 1").fetchone()
        if model is None:
            abort(404)
        metrics = conn.execute("SELECT * FROM Benchmark_Model_Metric WHERE model_id=? ORDER BY mae_tco2e",
                               (model["model_id"],)).fetchall()
        sims = conn.execute("""SELECT * FROM Credit_Simulation WHERE model_id=?
                               ORDER BY reduction_fraction, method""", (model["model_id"],)).fetchall()
    return render_template("benchmark_model.html", model=model, metrics=metrics, sims=sims,
                           limitations=model["limitations"].split("\n"))


def _trade_page_data(conn):
    establishments = conn.execute("SELECT est_id, est_name FROM Establishment ORDER BY est_name").fetchall()
    wallets = conn.execute("""
        SELECT cw.est_id, e.est_name, cw.available_credit, cw.reserved_credit
        FROM Credit_Wallet cw JOIN Establishment e ON cw.est_id = e.est_id
        ORDER BY e.est_name""").fetchall()
    transactions = conn.execute("""
        SELECT ct.transaction_id, s.est_name AS seller, b.est_name AS buyer, ct.credit_amount,
               ct.price_per_credit, ct.total_amount, ct.status, ct.created_at
        FROM Credit_Transaction ct
        JOIN Establishment s ON ct.seller_est_id = s.est_id
        JOIN Establishment b ON ct.buyer_est_id = b.est_id
        ORDER BY ct.created_at DESC LIMIT 3""").fetchall()
    listings = conn.execute("""
        SELECT cl.listing_id, cl.seller_est_id, e.est_name, cl.credit_amount, cl.remaining_credit,
               cl.price_per_credit, cl.status, cl.created_at
        FROM Credit_Listing cl JOIN Establishment e ON cl.seller_est_id = e.est_id
        WHERE cl.status = 'Active' AND cl.remaining_credit >= 0.01 - 1e-9
        ORDER BY cl.created_at DESC""").fetchall()
    stats = conn.execute("""
        SELECT COUNT(*) AS n, COALESCE(SUM(remaining_credit),0) AS credits,
               COALESCE(AVG(price_per_credit),0) AS avg_p, COALESCE(MAX(price_per_credit),0) AS hi,
               COALESCE(MIN(price_per_credit),0) AS lo
        FROM Credit_Listing WHERE status = 'Active'""").fetchone()
    total_trade_value = conn.execute(
        "SELECT COALESCE(SUM(total_amount),0) FROM Credit_Transaction WHERE status='Completed'"
    ).fetchone()[0]
    price_history = conn.execute("""
        SELECT created_at, price_per_credit, credit_amount, total_amount
        FROM Credit_Transaction WHERE status='Completed' ORDER BY created_at ASC""").fetchall()
    daily_market = conn.execute("""
        SELECT DATE(created_at, 'localtime') AS trade_date,
               ROUND(SUM(total_amount) / NULLIF(SUM(credit_amount), 0), 2) AS avg_price,
               SUM(credit_amount) AS credits_traded, ROUND(SUM(total_amount), 2) AS trade_value
        FROM Credit_Transaction WHERE status='Completed'
        GROUP BY DATE(created_at, 'localtime') ORDER BY trade_date ASC""").fetchall()

    market_prices = [r["avg_price"] for r in daily_market]
    moving_average = []
    for i in range(len(market_prices)):
        window = market_prices[max(0, i - 2): i + 1]
        moving_average.append(round(sum(window) / len(window), 2))
    current_price = market_prices[-1] if market_prices else 0
    previous_price = market_prices[-2] if len(market_prices) > 1 else current_price
    price_change = ((current_price - previous_price) / previous_price * 100) if previous_price else 0
    market_trend = "UP" if price_change > 1 else "DOWN" if price_change < -1 else "STABLE"

    return dict(
        establishments=establishments, wallets=wallets, listings=listings,
        active_listings=stats["n"], credits_available=stats["credits"], average_price=stats["avg_p"],
        highest_price=stats["hi"], lowest_price=stats["lo"], total_trade_value=total_trade_value,
        price_history=price_history, price_dates=[r["created_at"] for r in price_history],
        price_values=[r["price_per_credit"] for r in price_history],
        market_dates=[r["trade_date"] for r in daily_market], market_prices=market_prices,
        current_price=current_price, price_change=price_change, market_trend=market_trend,
        moving_average=moving_average, prediction=get_price_prediction(), transactions=transactions)


@app.route("/trade", methods=["GET", "POST"])
def trade():
    if request.method == "POST":
        try:
            seller = to_int(request.form.get("seller_est_id"), "Seller")
            buyer = to_int(request.form.get("buyer_est_id"), "Buyer")
            amount = to_credits(request.form.get("credit_amount"), "Credit amount")
            price = to_float(request.form.get("price_per_credit"), "Price", non_negative=True)
            if seller == buyer:
                raise ValueError("Seller and buyer cannot be the same establishment.")
        except ValueError as e:
            return error(str(e))
        try:
            with write_transaction() as w:
                wallet = w.execute(
                    "SELECT available_credit, reserved_credit FROM Credit_Wallet WHERE est_id=?",
                    (seller,)).fetchone()
                if wallet is None or not w.execute("SELECT 1 FROM Credit_Wallet WHERE est_id=?",
                                                   (buyer,)).fetchone():
                    raise ValueError("Seller or buyer wallet not found.")
               
                if wallet["available_credit"] - wallet["reserved_credit"] < amount - EPS:
                    raise ValueError("Seller does not have enough unreserved credits.")
                w.execute("""INSERT INTO Credit_Transaction
                             (seller_est_id, buyer_est_id, credit_amount, price_per_credit, total_amount)
                             VALUES (?,?,?,?,?)""", (seller, buyer, amount, price, amount * price))
                update_market_history(w)
        except (ValueError, sqlite3.IntegrityError) as e:
            return error(f"Transaction failed: {e}")
        return redirect("/trade")

    with closing(get_db()) as conn:
        return render_template("trade.html", **_trade_page_data(conn))


@app.route("/market")
def market():
    with closing(get_db()) as conn:
        market_trend = get_market_trend(conn)
        market_data = conn.execute("SELECT * FROM Market_History ORDER BY trade_date DESC").fetchall()
    return render_template("market.html", market_data=market_data,
                           prediction=get_price_prediction(), market_trend=market_trend)


@app.route("/buy-credit", methods=["POST"])
def buy_credit():
    try:
        listing_id = to_int(request.form.get("listing_id"), "Listing")
        buyer = to_int(request.form.get("buyer_est_id"), "Buyer")
        quantity = to_credits(request.form.get("quantity"), "Quantity")
    except ValueError as e:
        return error(str(e))
    try:
        with write_transaction() as w:
            listing = w.execute("""SELECT seller_est_id, price_per_credit FROM Credit_Listing
                                   WHERE listing_id=?""", (listing_id,)).fetchone()
            if listing is None:
                return error("Listing not found", 404)
            if not w.execute("SELECT 1 FROM Credit_Wallet WHERE est_id=?", (buyer,)).fetchone():
                return error("Buyer not found", 404)
            if buyer == listing["seller_est_id"]:
                return error("Seller cannot buy their own listing")
            
            cur = w.execute("""
                UPDATE Credit_Listing
                SET remaining_credit = ROUND(remaining_credit - ?, 6),
                    status = CASE WHEN remaining_credit - ? <= ? THEN 'Sold' ELSE 'Active' END
                WHERE listing_id = ? AND status = 'Active' AND remaining_credit >= ? - ?""",
                (quantity, quantity, EPS, listing_id, quantity, EPS))
            if cur.rowcount != 1:
                return error("This listing is no longer active or has too few credits left")
            w.execute("UPDATE Credit_Listing SET remaining_credit = MAX(remaining_credit, 0) "
                      "WHERE listing_id = ?", (listing_id,))
            cur = w.execute("""UPDATE Credit_Wallet
                               SET reserved_credit = ROUND(reserved_credit - ?, 6), updated_at = CURRENT_TIMESTAMP
                               WHERE est_id = ? AND reserved_credit >= ? - ?""",
                            (quantity, listing["seller_est_id"], quantity, EPS))
            if cur.rowcount != 1:
                raise ValueError("Seller reservation is inconsistent; trade cancelled")
            w.execute("""UPDATE Credit_Wallet SET reserved_credit = MAX(reserved_credit, 0)
                         WHERE est_id = ?""", (listing["seller_est_id"],))
            
            w.execute("""INSERT INTO Credit_Transaction
                         (seller_est_id, buyer_est_id, credit_amount, price_per_credit, total_amount)
                         VALUES (?,?,?,?,?)""",
                      (listing["seller_est_id"], buyer, quantity, listing["price_per_credit"],
                       quantity * listing["price_per_credit"]))
            close_dust_listings(w)
            update_market_history(w)
    except (ValueError, sqlite3.DatabaseError) as e:
        return error(str(e))
    return redirect("/trade")


@app.route("/create-listing", methods=["POST"])
def create_listing():
    try:
        seller = to_int(request.form.get("seller_est_id"), "Seller")
        amount = to_credits(request.form.get("credit_amount"), "Credit amount")
        price = to_float(request.form.get("price_per_credit"), "Price", positive=True)
    except ValueError as e:
        return error(str(e))
    try:
        with write_transaction() as w:
            wallet = w.execute("""SELECT available_credit, reserved_credit FROM Credit_Wallet
                                  WHERE est_id = ?""", (seller,)).fetchone()
            if wallet is None:
                return error("Seller wallet not found", 404)
            if wallet["available_credit"] - wallet["reserved_credit"] < amount - EPS:
                return error("Insufficient unreserved credits")
            w.execute("""INSERT INTO Credit_Listing
                         (seller_est_id, credit_amount, remaining_credit, price_per_credit)
                         VALUES (?,?,?,?)""", (seller, amount, amount, price))
            w.execute("""UPDATE Credit_Wallet
                         SET reserved_credit = ROUND(reserved_credit + ?, 6), updated_at = CURRENT_TIMESTAMP
                         WHERE est_id = ?""", (amount, seller))
    except sqlite3.DatabaseError as e:
        return error(str(e))
    return redirect("/trade")


@app.route("/transactions")
def transactions():
    with closing(get_db()) as conn:
        rows = conn.execute("""
            SELECT ct.transaction_id, s.est_name AS seller, b.est_name AS buyer, ct.credit_amount,
                   ct.price_per_credit, ct.total_amount, ct.status, ct.created_at
            FROM Credit_Transaction ct
            JOIN Establishment s ON ct.seller_est_id = s.est_id
            JOIN Establishment b ON ct.buyer_est_id = b.est_id
            ORDER BY ct.created_at DESC""").fetchall()
    return render_template("transactions.html", transactions=rows)


@app.route("/chatbot", methods=["GET"])
def chatbot_page():
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST"])
def chatbot_api():
    if not request.is_json:
        return jsonify({"error": "Request must contain JSON."}), 400

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON request."}), 400

    question = data.get("question")

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Please enter a question."}), 400

    question = question.strip()

    if len(question) > 1000:
        return jsonify({
            "error": "Your question must be 1000 characters or fewer."
        }), 400

    try:
        answer = answer_question(question)
        return jsonify({"answer": answer}), 200

    except ChatbotConfigurationError:
        app.logger.exception("CarbonX chatbot configuration error")
        return jsonify({
            "error": "The AI assistant is not configured correctly."
        }), 503

    except ChatbotProviderError:
        app.logger.exception("CarbonX chatbot provider error")
        return jsonify({
            "error": "The AI service is temporarily unavailable. Please try again."
        }), 502


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")