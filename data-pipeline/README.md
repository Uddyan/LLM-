# Data Pipeline

This directory contains the data preprocessing pipeline components for the Neighborly AI platform.

## Structure

- **ingestion/**: Data ingestion layer
  - Apache Kafka / AWS Kinesis for real-time streaming
  - Apache Airflow for batch processing
  - AWS Lambda for event-driven ingestion

- **processing/**: Data cleaning and transformation
  - Remove duplicates, handle missing values
  - Standardize formats across brands
  - Entity resolution
  - PII detection and masking

- **embeddings/**: Embedding generation
  - Document chunking (512-1024 tokens per chunk)
  - Embedding generation using text-embedding models
  - Batch processing for efficiency

## Technologies

- Apache Kafka/Kinesis
- Apache Airflow/Prefect
- Apache Spark
- dbt (data build tool)
- Great Expectations
