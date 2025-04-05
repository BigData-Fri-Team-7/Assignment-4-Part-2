# Building a RAG Pipeline with Airflow

This repository contains the code for **Assignment 4.2: Building a RAG Pipeline with Airflow**. The project aims to develop an AI-powered information retrieval application that processes unstructured data from PDFs (e.g., NVIDIA quarterly reports) and web pages by implementing a Retrieval-Augmented Generation (RAG) pipeline. This pipeline is designed for modularity and extensibility to support future applications.

##Frontend:http://34.67.213.206:8501
##Backend:http://34.67.213.206:8000
##Demo Video:https://drive.google.com/file/d/1TnzO4GA__hnNtz6I7bBsRu4uFI5QIlFp/view?usp=sharing
## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Deployment](#deployment)
- [Contributing & Attestation](#contributing--attestation)
- [Documentation](#documentation)
- [AI Use Disclosure](#ai-use-disclosure)
- [License](#license)

## Project Overview

In this project, we:
- **Extract Data:** Retrieve NVIDIA quarterly reports for the past five years.
- **Orchestrate Workflows:** Use Apache Airflow to automate data ingestion, processing, and retrieval workflows.
- **Parse PDFs:** Implement three PDF parsing strategies:
  - Extend our Assignment 1 extraction capabilities.
  - Use [Docling](https://github.com/docling) for PDF parsing.
  - Leverage [Mistral OCR](https://mistral.ai/news/mistral-ocr) for improved text extraction.
- **Build the RAG Pipeline:** 
  - Implement a naive RAG system with manual embeddings and cosine similarity.
  - Integrate with vector databases such as Pinecone and ChromaDB.
  - Utilize at least three chunking strategies (recursive, semantic, and sliding window) to optimize retrieval.
  - Enable hybrid search to allow querying of specific quarter data for contextual answers.
- **User Interface & Testing:** 
  - Develop a Streamlit application for PDF uploads, parser selection, and querying.
  - Use FastAPI as the backend to handle PDF processing and question answering.
  - Integrate your preferred LLM to generate responses.
- **Deployment:** Create Docker pipelines for:
  - The Airflow-based data ingestion, processing, and retrieval workflow.
  - The Streamlit + FastAPI-based user interaction and querying pipeline.

## Features

- **Data Extraction & Parsing (25%)**  
  - Retrieve and parse NVIDIA quarterly reports.
  - Utilize Docling and Mistral OCR for PDF conversion.

- **RAG Implementation & Chunking Strategies (40%)**  
  - Implement naive RAG with manual embeddings.
  - Integration with Pinecone and ChromaDB.
  - Support recursive, semantic, and sliding window chunking methods.

- **User Interface & Integration**  
  - FastAPI backend to serve processed data.
  - Streamlit frontend for PDF upload, parser selection, and question answering.

- **Deployment & Dockerization**  
  - Docker pipelines for both Airflow workflows and the Streamlit/FastAPI application.

- **Documentation & Presentation**  
  - Comprehensive project documentation, diagrams, a CodeLab, and a demo video.

## Architecture Diagram:
![image](https://github.com/user-attachments/assets/e4793780-97ec-41b1-8905-556dd2cbd70c)


