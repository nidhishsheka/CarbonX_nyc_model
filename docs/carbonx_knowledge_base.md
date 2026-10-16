# CarbonX — Chatbot Knowledge Base

## 1. Project Overview

CarbonX is a carbon-accounting application with peer-emissions benchmarking and a simulated carbon-credit marketplace.

Its main purpose is to organize establishment activity data, emissions records, benchmark comparisons, carbon-credit information, and marketplace transactions in one application.

The project uses Python and Flask for its web application and SQLite for database storage. Its supporting modules include emissions benchmarking and a separate next-day credit-price forecasting model.

## 2. Main Features

### 2.1 Establishment Management

CarbonX maintains establishment records. The `Establishment` table includes the establishment ID, name, type, and location.

The application also stores activity information, such as the source, reporting period, quantity, and provenance associated with an activity.

### 2.2 Emissions Accounting

The database includes tables for:
- Activity data
- Emission sources and emission factors
- Calculated emission records
- Baseline emissions
- Allowed emission limits
- Data provenance

The `Emission_Record` table stores an emission amount in kilograms alongside references to its activity and emission factor.

Conceptually, emissions accounting converts activity measurements into estimated emissions using applicable emission factors. The precise formula, units, validation rules, and calculation sequence must be confirmed from the implementation before being presented as the application's exact method.

### 2.3 Peer Emissions Benchmarking

CarbonX includes a benchmarking module that predicts expected emissions and compares them with reported emissions.

The database stores:
- Benchmark model metadata
- Model evaluation metrics
- Expected emissions
- Lower and upper prediction bounds
- Reported emissions
- Comparison results and review actions
- Input and imputation information

The project README states that the benchmark was trained using New York City buildings data for report year 2024. It has not been validated for Indian establishments.

The nominal 90% prediction interval covered 87.7% of test properties in the reported evaluation. The evaluation used one fixed split and one year; repeated splits and future-year testing remain to be done.

Benchmark predictions are estimates and should not automatically be treated as proof of inaccurate reporting or regulatory non-compliance.

### 2.4 Carbon-Credit Marketplace

CarbonX includes database tables for:
- Carbon-credit records
- Credit wallets
- Credit listings
- Credit transactions
- Credit ledger entries
- Marketplace history

The database records information such as available and reserved credits, listing quantities, prices, transaction amounts, and transaction status.

The chatbot may explain these concepts and, once connected to approved read-only database functions, provide current marketplace information.

The chatbot must not create listings, buy credits, transfer credits, or modify balances unless a separately authorized and explicitly designed application workflow permits that action.

### 2.5 Credit Simulation and Reduction Policies

CarbonX includes reduction-policy records and credit-simulation results.

According to the project README, credits are simulated using a baseline and sector reduction assumptions in relation to recorded emissions. Seven seeded sectors—Mining, Food, Electronics, Petrochemical, Glass, Shipping, and Agriculture—have no reduction policy and are treated as having a 0% reduction rate; reports flag these sectors.

These credits are simulated and must not be described as independently verified, certified, or automatically tradable real-world environmental credits.

### 2.6 Electricity Data and Provenance

The database includes Central Electricity Authority (CEA) plant information, yearly plant records, yearly summaries, emission factors, and provenance metadata.

The project README states that the CEA version 22.0 source is used for the grid-electricity factor. Coal, diesel, natural-gas, and petrol factors are still unsourced seed values.

The application should disclose these source limitations when users ask about the reliability of emission calculations.

## 3. Machine-Learning Modules

### 3.1 Benchmarking Model

The benchmarking module predicts expected emissions and provides an interval around the prediction. Its database metadata includes the algorithm, interval method, coverage information, model metrics, and limitations.

The exact prediction pipeline should be explained using the source code and model metadata.

### 3.2 Credit-Price Forecasting

The `ml_model.py` module forecasts the next day's credit price. According to the project README, this is a market model and is separate from the emissions-benchmarking model.

A price forecast is an estimate, not a guaranteed future market price.

### 3.3 Separate Steel Experiment

The project README describes a steel time-series experiment in the `research/` directory. It uses an illustrative factor of 0.5 kgCO2e/kWh and is not connected to the main application.

The chatbot must not claim that this experiment drives the application's live emissions calculations.

## 4. Database Reference

The following tables are identified in the database schema.

| Area | Tables |
|---|---|
| Establishments | `Establishment`, `Establishment_Profile`, `Establishment_Policy` |
| Activity and emissions | `Activity_Data`, `Emission_Source`, `Emission_Factor`, `Emission_Record` |
| Baselines and policies | `Baseline_Emission`, `Allowed_Limit`, `Reduction_Policy` |
| Benchmarking | `Benchmark_Model`, `Benchmark_Model_Metric`, `Benchmark_Prediction` |
| Credit marketplace | `Carbon_Credit`, `Credit_Wallet`, `Credit_Listing`, `Credit_Transaction`, `Credit_Ledger`, `Market_History` |
| Credit simulation | `Credit_Simulation` |
| Data sources | `Data_Provenance`, `CEA_Plant`, `CEA_Plant_Year`, `CEA_Year_Summary` |

The database schema describes the available columns and keys. The actual application code must be consulted to confirm business rules, relationships, calculation formulas, and which tables are actively used by each feature.

## 5. Chatbot Responsibilities

The CarbonX chatbot should:

1. Explain the application's features and workflows in beginner-friendly language.
2. Answer questions using this knowledge document and other approved project documentation.
3. Explain emissions accounting, benchmarking, model uncertainty, and credit-marketplace terminology.
4. Retrieve current database facts only through controlled, read-only backend functions.
5. Distinguish stored facts, model estimates, simulated values, and general explanations.
6. Explain relevant project limitations when answering questions about reliability.
7. Say when the available sources do not contain enough information to answer confidently.
8. Avoid exposing private establishment data to unauthorized users.

## 6. Grounding and Safety Rules

- Do not invent emission factors, calculation formulas, policy requirements, or database values.
- Do not present a prediction as a guaranteed result.
- Do not describe simulated credits as verified environmental credits.
- Do not assume that every database table contains current records.
- Do not generate or execute arbitrary SQL from a user's natural-language request.
- Do not allow the language model to directly modify credit balances or marketplace transactions.
- Use approved database functions and the application's existing authorization checks for live information.
- When possible, identify the documentation section or database-backed source used in an answer.
- If a question concerns information not documented or available through an approved function, clearly state that limitation.

## 7. Known Project Limitations

1. The benchmarking model uses New York City building data for report year 2024 and is not validated for Indian establishments.
2. Its reported nominal 90% interval covered 87.7% of test properties.
3. The benchmark evaluation used one fixed data split and one year.
4. Credit quantities are simulated.
5. Seven seeded sectors have no reduction policy and are treated as having a 0% reduction rate.
6. Coal, diesel, natural-gas, and petrol emission factors are unsourced seed values.
7. The steel time-series experiment is separate from the main application.
8. Evidence collection, additionality assessment, and verifier sign-off before credits become tradable are identified as future work.

## 8. Scope of This Document

This document is an initial project knowledge source for the CarbonX chatbot. It should be updated as application workflows, source code, database behavior, and supporting documentation are reviewed.

Where the document does not specify an implementation detail, the chatbot must not treat an assumption as an established project fact.